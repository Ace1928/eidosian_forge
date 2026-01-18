import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import argparse
import os
from filelock import FileLock
import tempfile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from ray.tune.examples.pbt_dcgan_mnist.common import (

Example of training DCGAN on MNIST using PBT with Tune's function API.
