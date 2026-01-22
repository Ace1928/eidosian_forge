from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
@dataclass
class LayerwiseMemoryTrackerSummary:
    """
    Summary of the memory allocation during forward/backward
    - max_memory_allocated: the peak of memory allocated
    - max_memory_cached: the peak of memory cached by PyTorch
    - total_activation_allocations: cumulative count of activations allocations
    - total_forward_allocations: cumulative count of forward pass allocations
    - top_forward_activation_producers: layers that allocated the most activations
    """
    max_memory_allocated: int
    max_memory_cached: int
    total_activation_allocations: int
    total_forward_allocations: int
    top_forward_activation_producers: List[LayerMemoryTrace]