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
def _init_shard_metadata(self, numel_padded: int, unsharded_start_idx: int, unsharded_end_idx: int) -> None:
    """
        Initializes shard-related metadata for this rank's shard of the flat
        parameter: ``_sharded_size``, ``_shard_param_infos``, and
        ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
    flat_param = self.flat_param
    flat_param._sharded_size = flat_param.size()
    sharded_flat_param_numel = flat_param.numel()
    _p_assert(unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx, f'unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}')
    _p_assert(numel_padded <= sharded_flat_param_numel, f'numel_padded: {numel_padded} sharded_flat_param_numel: {sharded_flat_param_numel}')
    shard_param_infos = self._get_shard_metadata(unsharded_start_idx, unsharded_end_idx)
    assert len(shard_param_infos) == flat_param._num_params, f'Expects length {flat_param._num_params} but got {len(shard_param_infos)}'
    flat_param._shard_param_infos = shard_param_infos
    flat_param._shard_numel_padded = numel_padded