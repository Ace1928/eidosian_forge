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
def _reset_flat_param_grad_info_if_needed(self):
    """
        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
    if not self._use_orig_params:
        return
    flat_param = self.flat_param
    assert flat_param._params is not None
    all_grad_none = True
    requires_grad = False
    for param in flat_param._params:
        all_grad_none &= param.grad is None
        requires_grad |= param.requires_grad
    if all_grad_none:
        flat_param.grad = None
    flat_param.requires_grad = requires_grad