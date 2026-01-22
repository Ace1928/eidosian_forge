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
@property
def _fqns_in_shard(self) -> List[str]:
    """Returns the FQNs of the parameters present in this rank's shard."""
    fqns_in_shard: List[str] = []
    for fqn, shard_param_info in zip(self.flat_param._fqns, self.flat_param._shard_param_infos):
        if shard_param_info.in_shard:
            fqns_in_shard.append(fqn)
    return fqns_in_shard