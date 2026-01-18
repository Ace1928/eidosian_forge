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
@contextlib.contextmanager
def to_cpu(self):
    """
        Moves the unpadded unsharded flat parameter to CPU while in the context
        and moves it back to the previous device upon exit. For now, this
        assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
    self._check_sharded_strategy()
    _p_assert(self.flat_param.size() == self.flat_param._unpadded_unsharded_size, f'Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}')
    self._check_on_compute_device(self.flat_param)
    unpadded_storage_ptr = self.flat_param._typed_storage()._data_ptr()
    padded_storage_ptr = self._get_padded_unsharded_flat_param()._typed_storage()._data_ptr()
    _p_assert(unpadded_storage_ptr == padded_storage_ptr, 'Expects the unpadded parameter to be a view into the padded parameter')
    self.flat_param_to(torch.device('cpu'))
    self._free_unsharded_flat_param()
    try:
        yield
    finally:
        _p_assert(self.flat_param.size() == self.flat_param._unpadded_unsharded_size, f'Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}')
        padded_unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param[:self.flat_param.numel()].copy_(self.flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)