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
@classmethod
def _split_forward_backward(cls, memory_traces: List[LayerMemoryTrace], values: List[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.array(list(range(len(memory_traces))))
    mask_forwards, mask_backwards = cls._mask_forward_backward(memory_traces)
    return (x_values, np.ma.masked_where(mask_backwards, values), np.ma.masked_where(mask_forwards, values))