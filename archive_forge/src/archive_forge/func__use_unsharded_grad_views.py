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
def _use_unsharded_grad_views(self) -> None:
    """
        Unflattens the unsharded flat parameter's gradient by setting the
        original parameter variables' gradients to be views into it.
        """
    if self.flat_param.grad is None:
        for param in chain(self.flat_param._params, self.flat_param._shared_params):
            param.grad = None
        return
    self._check_unsharded(self.flat_param.grad)
    views = self._get_unflat_views(self.flat_param.grad)
    for i, (view, (param_name, module, _)) in enumerate(zip(views, self.flat_param._param_infos)):
        _p_assert(hasattr(module, param_name), f'{self.flat_param._fqns[i]} is missing')
        param = getattr(module, param_name)
        if param.shape != view.shape or param.dtype != view.dtype or param.device != view.device:
            if param.grad is None:
                param.grad = torch.empty_like(param)
            param.grad.data = view
        else:
            param.grad = view
    for i, (param_name, module, module_name, prim_param_name, prim_module, _) in enumerate(self.flat_param._shared_param_infos):
        _p_assert(hasattr(module, param_name), f'{(module_name + '.' + param_name if module_name else param_name)} is missing')
        param = getattr(module, param_name)
        prim_param = getattr(prim_module, prim_param_name)
        if param.shape != prim_param.grad.shape or param.dtype != prim_param.grad.dtype or param.device != prim_param.grad.device:
            if param.grad is None:
                param.grad = torch.empty_like(param)
            param.grad.data = prim_param.grad
        else:
            param.grad = prim_param.grad