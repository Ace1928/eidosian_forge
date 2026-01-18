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
def unshard(self):
    """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
    if not self.needs_unshard():
        unsharded_flat_param = self._get_padded_unsharded_flat_param() if self.uses_sharded_strategy else self.flat_param
        self._use_unsharded_flat_param(unsharded_flat_param)
        return
    unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
    padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
    self._use_unsharded_flat_param(padded_unsharded_flat_param)