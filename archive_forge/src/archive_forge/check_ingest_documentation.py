import sys
import time
from typing import Optional
import numpy as np
import ray
from ray import train
from ray.air.config import DatasetConfig, ScalingConfig
from ray.data import Dataset, DataIterator, Preprocessor
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import DataConfig
from ray.util.annotations import Deprecated, DeveloperAPI
Make a debug train loop that runs for the given amount of epochs.