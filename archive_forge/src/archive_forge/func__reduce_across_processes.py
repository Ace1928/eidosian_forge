import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def _reduce_across_processes(tensor, reduction='mean', scale=1.0):
    state = PartialState()
    cloned_tensor = tensor.clone()
    if state.distributed_type == DistributedType.NO:
        return cloned_tensor
    if state.distributed_type == DistributedType.XLA:
        xm.mark_step()
        xm.all_reduce(xm.REDUCE_SUM, [cloned_tensor], scale)
        xm.mark_step()
    elif state.distributed_type.value in TORCH_DISTRIBUTED_OPERATION_TYPES:
        torch.distributed.all_reduce(cloned_tensor, ReduceOp.SUM)
    if reduction == 'mean':
        cloned_tensor /= state.num_processes
    return cloned_tensor