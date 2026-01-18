import argparse
import os
import tempfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import ray
import ray.cloudpickle as cpickle
from ray import train, tune
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
def update_optimizer_config(optimizer, config):
    for param_group in optimizer.param_groups:
        for param, val in config.items():
            param_group[param] = val