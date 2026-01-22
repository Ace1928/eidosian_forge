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
def _alloc_padded_unsharded_flat_param(self):
    """
        Allocates the *padded* unsharded flat parameter. The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
    self._check_sharded_strategy()
    flat_param = self.flat_param
    unsharded_flat_param = self._get_padded_unsharded_flat_param()
    self._check_storage_freed(unsharded_flat_param)
    _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)
    return unsharded_flat_param