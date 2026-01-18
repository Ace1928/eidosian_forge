import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from ray.tune.examples.mnist_pytorch import train_func, test_func, ConvNet,\
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
Train a Pytorch ConvNet with Trainable and PopulationBasedTraining
       scheduler. The example reuse some of the functions in mnist_pytorch,
       and is a good demo for how to add the tuning function without
       changing the original training code.
    