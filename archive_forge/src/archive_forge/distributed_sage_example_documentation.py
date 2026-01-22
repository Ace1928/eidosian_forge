import argparse
import os
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch_geometric.datasets import FakeDataset, Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomNodeSplit
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
Returns a function to be called on each worker that returns Reddit Dataset.