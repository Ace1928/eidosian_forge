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
def _collect_tensors(module_io_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    """
        Extract the tensors out of the provided input or output of a nn.Module
        """
    tensors = []
    to_visit = [module_io_tensors]
    while to_visit:
        x = to_visit.pop()
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        elif isinstance(x, tuple) or isinstance(x, list):
            to_visit.extend(module_io_tensors)
    return tensors