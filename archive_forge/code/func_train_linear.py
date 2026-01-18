import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
def train_linear(num_workers=1, num_hidden_layers=1, use_auto_transfer=True, epochs=3):
    config = {'lr': 0.01, 'hidden_size': num_hidden_layers, 'batch_size': 4096, 'epochs': epochs, 'use_auto_transfer': use_auto_transfer}
    trainer = TorchTrainer(train_func, train_loop_config=config, scaling_config=ScalingConfig(use_gpu=True, num_workers=num_workers))
    results = trainer.fit()
    print(results.metrics)
    return results