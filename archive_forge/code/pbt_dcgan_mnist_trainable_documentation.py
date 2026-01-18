import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
import argparse
import os
from filelock import FileLock
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from common import beta1, MODEL_PATH
from common import demo_gan, get_data_loader, plot_images, train_func, weights_init
from common import Discriminator, Generator, Net

Example of training DCGAN on MNIST using PBT with Tune's Trainable Class
API.
