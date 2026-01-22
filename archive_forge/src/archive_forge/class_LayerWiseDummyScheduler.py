import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import (
class LayerWiseDummyScheduler(LRScheduler):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization and scheduling step
    are already done through the post gradient hooks. Therefore
    the trick is to create a dummy scheduler that can take arbitrary
    args and kwargs and return a no-op during training.
    """

    def __init__(self, *args, **kwargs):
        optimizer = LayerWiseDummyOptimizer()
        last_epoch = -1
        verbose = False
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs