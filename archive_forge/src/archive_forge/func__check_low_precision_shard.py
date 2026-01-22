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
def _check_low_precision_shard(self):
    _p_assert(self._uses_param_mixed_precision, 'Not using low precision for parameters')
    _p_assert(getattr(self.flat_param, '_mp_shard', None) is not None, 'Expects `_mp_shard` to exist')
    device = self.flat_param._mp_shard.device
    _p_assert(device == self.device, f'Expects the low precision shard to be on {self.device} but got {device}')