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
@staticmethod
def _is_same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
        Indicate if x and y share the same storage, meaning that one of them
        is a view, reshape or stride of the other or from a common tensor
        """
    return x.storage().data_ptr() == y.storage().data_ptr()