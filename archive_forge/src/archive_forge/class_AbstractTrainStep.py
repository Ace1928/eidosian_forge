import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
class AbstractTrainStep(ABC):
    """Abstract class for batching, forward pass and loss handler."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_batch_func(self):
        pass

    def get_forward_step_func(self):
        pass

    def get_loss_func(self):
        pass