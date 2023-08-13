import os
import json
import pyarrow.parquet as pq
import torch
import torch.optim as optim
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

json_folder = 'json_fold'


def load_data_from_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def load_data_from_parquet(parquet_path):
    return pq.read_table(parquet_path)


def split_text_into_portions(text, portion_length):
    return [text[i:i + portion_length] for i in range(0, len(text), portion_length)]


json_data = []
for json_file in os.listdir(json_folder):
    if json_file.endswith('.json'):
        json_data.append(load_data_from_json(os.path.join(json_folder, json_file)))

parquet_data = []
for parquet_file in os.listdir(json_folder):
    if parquet_file.endswith('.parquet'):
        parquet_data.append(load_data_from_parquet(os.path.join(json_folder, parquet_file)))

all_data = json_data + parquet_data
all_portions = []

for item in all_data:
    portions = split_text_into_portions(item['json_fold'], 5000)
    all_portions.extend(portions)


def load_model_and_tokenizer(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
model, tokenizer = load_model_and_tokenizer(model_name)

num_epochs = 10
lr = 1e-3
output_dir = 'trained_model'

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for training!")
    model = nn.DataParallel(model)

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

optimizer = optim.AdamW(model.parameters(), lr=lr)

portion_num = 0

for epoch in range(num_epochs):

    print(f"Эпоха {epoch + 1}")

    epoch_loss = 0

    start_time = time.time()

    for portion in portions:
        input_ids = tokenizer.encode(portion, return_tensors='pt', truncation=True)

        output = model(input_ids, labels=input_ids)
        loss = output[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        portion_num += 1

        print(f"Порция {portion_num} - Потери: {loss.item()}")

    end_time = time.time()
    epoch_time = end_time - start_time

    avg_loss = epoch_loss / len(portions)

    print(f"Средние потери эпохи: {avg_loss}")
    print(f"Время эпохи: {epoch_time} секунд")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
