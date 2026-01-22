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
def _get_shard_metadata(self, unsharded_start_idx: int, unsharded_end_idx: int) -> Tuple[_ShardParamInfo, ...]:
    """
        Computes the shard metadata based on ``unsharded_start_idx`` and
        ``unsharded_end_idx`` (inclusive), which give the interval of the
        unsharded flat parameter specifying the shard.
        """
    flat_param_offsets = self._get_flat_param_offsets()
    assert len(flat_param_offsets) == len(self.flat_param._numels_with_padding), f'Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}'
    shard_param_infos: List[_ShardParamInfo] = []
    sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
    for i, ((unsharded_param_start_idx, unsharded_param_end_idx), is_padding) in enumerate(zip(flat_param_offsets, self.flat_param._is_padding_mask)):
        if is_padding:
            continue
        in_sharded_flat_param = unsharded_start_idx <= unsharded_param_end_idx and unsharded_end_idx >= unsharded_param_start_idx
        if not in_sharded_flat_param:
            shard_param_info = _ShardParamInfo(False, None, None, None, None)
        else:
            if unsharded_start_idx <= unsharded_param_start_idx:
                intra_param_start_idx = 0
                offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
            else:
                intra_param_start_idx = unsharded_start_idx - unsharded_param_start_idx
                offset_in_shard = 0
            assert offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel, f'Invalid `offset_in_shard` of {offset_in_shard} for sharded flat parameter with {sharded_flat_param_numel} numel'
            intra_param_end_idx = min(unsharded_param_end_idx, unsharded_end_idx) - unsharded_param_start_idx
            numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
            shard_param_info = _ShardParamInfo(True, offset_in_shard, numel_in_shard, intra_param_start_idx, intra_param_end_idx)
        shard_param_infos.append(shard_param_info)
    return tuple(shard_param_infos)