import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
def train_epoch(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()