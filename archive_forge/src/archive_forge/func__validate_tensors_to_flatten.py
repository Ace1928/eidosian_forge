import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def _validate_tensors_to_flatten(self, tensors: List[Union[Tensor, nn.Parameter]]) -> Tuple:
    """
        Validates the tensors to flatten and returns any necessary metadata.
        """
    dtype: Optional[torch.dtype] = None
    flat_param_requires_grad: Optional[bool] = None
    device: Optional[torch.device] = None
    for tensor in tensors:
        if isinstance(tensor, FlatParameter):
            raise ValueError('Cannot flatten a `FlatParameter`')
        if dtype is None and (not tensor.is_floating_point()):
            raise ValueError('Cannot flatten integer dtype tensors')
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f'Must flatten tensors with uniform dtype but got {dtype} and {tensor.dtype}')
        if not self._use_orig_params and flat_param_requires_grad is not None and (tensor.requires_grad != flat_param_requires_grad):
            raise ValueError('Must flatten tensors with uniform `requires_grad` when `use_orig_params=False`')
        if device is not None and tensor.device != device:
            raise ValueError(f'Must flatten tensors on the same device but got both {device} and {tensor.device}')
        dtype = tensor.dtype
        flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
        device = tensor.device
    assert flat_param_requires_grad is not None, 'Requires non-empty `tensors` list'
    return (dtype, flat_param_requires_grad, device)