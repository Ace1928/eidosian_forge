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
def train_gnn(num_workers=2, use_gpu=False, epochs=3, global_batch_size=32, dataset='reddit'):
    per_worker_batch_size = global_batch_size // num_workers
    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker, train_loop_config={'num_epochs': epochs, 'batch_size': per_worker_batch_size, 'dataset_fn': gen_reddit_dataset if dataset == 'reddit' else gen_fake_dataset()}, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu))
    result = trainer.fit()
    print(result.metrics)