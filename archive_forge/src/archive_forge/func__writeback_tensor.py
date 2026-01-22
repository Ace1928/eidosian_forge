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
def _writeback_tensor(self, src_tensor: Optional[Tensor], dst_tensor: Tensor, tensor_index: int, expected_shape: torch.Size, offset: int, is_param: bool) -> None:
    """
        Writes back ``src_tensor`` to ``dst_tensor`` at offset ``offset``,
        where ``src_tensor`` should have shape ``expected_shape``. ``is_param``
        indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
    _p_assert(len(expected_shape) == 1, f'Expects a 1D expected shape but got {expected_shape}')
    if self._debug_level == dist.DebugLevel.INFO:
        rank = self.rank if hasattr(self, 'rank') else dist.get_rank()
        src_shape = src_tensor.shape if src_tensor is not None else None
        src_device = src_tensor.device if src_tensor is not None else None
        warnings.warn(f'[Rank {rank}] {('Parameter' if is_param else 'Gradient')} needs writeback in {self._training_state}\nexpected shape={expected_shape} shape={src_shape} expected device={dst_tensor.device} device={src_device}')
    if src_tensor is not None and src_tensor.shape != expected_shape:
        raise RuntimeError(f'Cannot writeback when the {('parameter' if is_param else 'gradient')} shape changes\nExpects {expected_shape} but got {src_tensor.shape}')
    if src_tensor is not None:
        dst_tensor[offset:offset + expected_shape.numel()].copy_(src_tensor)
    else:
        dst_tensor[offset:offset + expected_shape.numel()].zero_()
        assert self.flat_param._is_grad_none_mask is not None
        self.flat_param._is_grad_none_mask[tensor_index] = True