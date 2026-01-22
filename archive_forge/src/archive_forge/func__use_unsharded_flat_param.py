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
def _use_unsharded_flat_param(self, padded_unsharded_flat_param: torch.Tensor) -> None:
    """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.
        """
    unsharded_size = self.flat_param._unpadded_unsharded_size
    self.flat_param.data = padded_unsharded_flat_param[:unsharded_size.numel()].view(unsharded_size)
    in_forward = self._training_state == HandleTrainingState.FORWARD
    in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
    if self._use_orig_params:
        if self._skipped_use_sharded_views and in_pre_backward:
            return
        self._use_unsharded_views(as_params=not in_forward and (not in_pre_backward))
    elif in_forward:
        self._use_unsharded_views(as_params=False)