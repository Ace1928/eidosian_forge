import argparse
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
def validate_epoch(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
    loss /= num_batches
    import copy
    model_copy = copy.deepcopy(model)
    return (model_copy.cpu().state_dict(), loss)