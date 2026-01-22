import multiprocessing
import os
import sys
from functools import partial
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, no_type_check
from unittest.mock import Mock
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import select_topk, to_onehot
from torchmetrics.utilities.enums import DataType
def _allclose_recursive(res1: Any, res2: Any, atol: float=1e-06) -> bool:
    """Recursively asserting that two results are within a certain tolerance."""
    if isinstance(res1, Tensor):
        return torch.allclose(res1, res2, atol=atol)
    if isinstance(res1, str):
        return res1 == res2
    if isinstance(res1, Sequence):
        return all((_allclose_recursive(r1, r2) for r1, r2 in zip(res1, res2)))
    if isinstance(res1, Mapping):
        return all((_allclose_recursive(res1[k], res2[k]) for k in res1))
    return res1 == res2