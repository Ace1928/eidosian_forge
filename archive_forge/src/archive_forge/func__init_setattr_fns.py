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
def _init_setattr_fns(self):
    use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, '') == '1'
    self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
    self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
    if use_unsafe_setattr:
        self._setattr_tensor = _unsafe_setattr_tensor
        self._setattr_param = _unsafe_setattr_param
    else:
        self._setattr_tensor = _safe_setattr_tensor_or_param
        self._setattr_param = _safe_setattr_tensor_or_param