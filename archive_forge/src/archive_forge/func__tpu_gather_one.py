import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _tpu_gather_one(tensor):
    if tensor.ndim == 0:
        tensor = tensor.clone()[None]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return xm.all_gather(tensor)