import argparse
import os
import torch
import torch.utils.data
import torchvision
from filelock import FileLock
from torchvision import datasets, transforms
import ray
from ray import train
from ray.train import ScalingConfig
def train_mosaic_cifar10(num_workers=2, use_gpu=False, max_duration='5ep'):
    from composer.algorithms import LabelSmoothing
    from ray.train.mosaic import MosaicTrainer
    trainer_init_config = {'max_duration': max_duration, 'algorithms': [LabelSmoothing()], 'should_eval': False}
    trainer = MosaicTrainer(trainer_init_per_worker=trainer_init_per_worker, trainer_init_config=trainer_init_config, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu))
    result = trainer.fit()
    print(f'Results: {result.metrics}')
    return result