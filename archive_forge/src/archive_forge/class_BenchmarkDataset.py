import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
class BenchmarkDataset(torch.utils.data.Dataset):
    """Create a naive dataset for the benchmark"""

    def __init__(self, dim, size=1000):
        self.x = torch.from_numpy(np.random.normal(size=(size, dim))).float()
        self.y = torch.from_numpy(np.random.normal(size=(size, 1))).float()
        self.size = size

    def __getitem__(self, index):
        return (self.x[index, None], self.y[index, None])

    def __len__(self):
        return self.size