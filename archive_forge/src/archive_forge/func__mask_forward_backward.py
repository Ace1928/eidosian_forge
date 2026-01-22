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
def _mask_forward_backward(cls, memory_traces: List[LayerMemoryTrace]) -> Tuple[np.ndarray, np.ndarray]:
    mask_forwards = np.array([t.is_forward for t in memory_traces])
    return (mask_forwards, ~mask_forwards)