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
def _all_gather_flat_param(self, padded_unsharded_flat_param: Tensor) -> Tensor:
    """
        All-gathers the handle's flat parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
    _p_assert(hasattr(self, 'process_group') and hasattr(self, 'world_size'), 'Expects a process group and world size to have been set via `shard()`')
    sharded_flat_param = self.flat_param.data
    expected_numel = sharded_flat_param.numel() * self.world_size
    _p_assert(padded_unsharded_flat_param.numel() == expected_numel, f'Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}')
    if sharded_flat_param.is_cpu:
        tensor_list = list(torch.chunk(padded_unsharded_flat_param, dist.get_world_size(self.process_group)))
        work = dist.all_gather(tensor_list, sharded_flat_param, group=self.process_group)
    else:
        dist.all_gather_into_tensor(padded_unsharded_flat_param, sharded_flat_param, self.process_group)
    return padded_unsharded_flat_param