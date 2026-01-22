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
@no_type_check
def _get_unflat_views_aligned(self, tensor: Optional[Tensor]=None) -> List[Tensor]:
    """
        This has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
    flat_param = self.flat_param
    if tensor is None:
        tensor = flat_param
    splits: List[Tensor] = torch.split(tensor, flat_param._numels_with_padding, dim=0)
    idx = 0
    views: List[Tensor] = []
    for split, is_padding in zip(splits, flat_param._is_padding_mask):
        if is_padding:
            continue
        views.append(_ext_post_unflatten_transform(split.view(flat_param._shapes[idx]), flat_param._param_extensions[idx]))
        idx += 1
    return views