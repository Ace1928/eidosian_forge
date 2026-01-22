import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
@torch.no_grad()
def _writeback_to_local_shard(handle: FlatParamHandle, writeback_grad: bool):
    """
    For the handle, writes back the this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: The handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """

    def _get_shard(flat_param_or_grad: torch.Tensor) -> torch.Tensor:
        if handle.uses_sharded_strategy:
            shard, _ = FlatParamHandle._get_unpadded_shard(flat_param_or_grad, handle.rank, handle.world_size)
            return shard
        return flat_param_or_grad
    param_shard = _get_shard(handle.flat_param)
    handle.flat_param._local_shard[:param_shard.numel()].copy_(param_shard)
    if writeback_grad:
        existing_grad = handle.sharded_grad
        if existing_grad is not None:
            assert handle.flat_param.grad is not None
            grad_shard = _get_shard(handle.flat_param.grad)
            existing_grad[:grad_shard.numel()].copy_(grad_shard)