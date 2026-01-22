import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
Create a naive dataset for the benchmark