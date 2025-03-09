import torch
from torch import nn

def create_model():
    # Создаем простую модель с двумя полносвязными слоями
model = nn.Sequential(
    nn.Linear(10, 50),  # Входной слой: 10 входов, 50 выходов
    nn.ReLU(),          # Активационная функция ReLU
    nn.Linear(50, 1)    # Выходной слой: 50 входов, 1 выход
)
    return model

def count_parameters(model):
    # Подсчитываем все обучаемые параметры модели
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
