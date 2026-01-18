import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import test_func, ConvNet, get_data_loaders
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
Test the best model given output of tuner.fit().