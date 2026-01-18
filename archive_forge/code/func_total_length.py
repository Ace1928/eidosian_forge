import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
@property
def total_length(self):
    return len(self.batch_sampler)