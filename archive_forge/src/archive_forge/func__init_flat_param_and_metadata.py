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
def _init_flat_param_and_metadata(self, params: List[Union[Tensor, nn.Parameter]], module: nn.Module, aligned_numel: int, use_orig_params: bool) -> None:
    """
        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
    if len(params) == 0:
        raise ValueError('Expects non-empty `params`')
    if aligned_numel < 0:
        raise ValueError(f'Expects non-negative `aligned_numel` but got {aligned_numel}')
    dtype, flat_param_requires_grad, device = self._validate_tensors_to_flatten(params)
    params_set = set(params)
    param_infos: List[ParamInfo] = []
    numels: List[int] = []
    shapes: List[torch.Size] = []
    fqns: List[str] = []
    shared_param_infos: List[SharedParamInfo] = []
    shared_param_memo: Dict[Union[Tensor, nn.Parameter], Tuple[nn.Module, str, str]] = {}
    params_to_flatten: List[Union[Tensor, nn.Parameter]] = []
    shared_params: List[Union[Tensor, nn.Parameter]] = []
    param_extensions: List[Any] = []
    is_padding_mask: List[bool] = []
    total_numel = total_numel_without_padding = 0
    for submodule_name, submodule in module.named_modules(remove_duplicate=False):
        for param_name, param in _named_parameters_with_duplicates(submodule, recurse=False):
            if param not in params_set:
                continue
            if param in shared_param_memo:
                prim_module, prim_module_name, prim_param_name = shared_param_memo[param]
                shared_params.append(param)
                shared_param_infos.append(SharedParamInfo(param_name, submodule, submodule_name, prim_param_name, prim_module, prim_module_name))
            else:
                if aligned_numel > 0:
                    numel_to_pad = aligned_numel - total_numel % aligned_numel
                    if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                        padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                        params_to_flatten.append(padding_tensor)
                        is_padding_mask.append(True)
                        numels.append(numel_to_pad)
                        total_numel += numel_to_pad
                transform_t, extension = _ext_pre_flatten_transform(param)
                param = cast(nn.Parameter, transform_t)
                param_extensions.append(extension)
                shared_param_memo[param] = (submodule, submodule_name, param_name)
                params_to_flatten.append(param)
                is_padding_mask.append(False)
                param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                numels.append(param.numel())
                shapes.append(param.shape)
                fqn = submodule_name + '.' + param_name if submodule_name else param_name
                fqns.append(fqn)
                total_numel += param.numel()
                total_numel_without_padding += param.numel()
    if len(params_to_flatten) == 0:
        raise ValueError(f"`params` were not found in `module`'s treeparams: {params}\nmodule: {module}")
    if self.rank == 0 and aligned_numel > 0 and (total_numel != total_numel_without_padding):
        log.info('FSDP FlatParameter address alignment created %s numel of padding (%s vs. %s)', total_numel - total_numel_without_padding, total_numel, total_numel_without_padding)
    if aligned_numel > 0:
        numel_to_pad = self.world_size - total_numel % self.world_size
        if numel_to_pad > 0 and numel_to_pad < self.world_size:
            if self.rank == 0:
                log.info('FSDP FlatParameter world size divisibility created %s numel of padding', numel_to_pad)
            padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
            params_to_flatten.append(padding_tensor)
            is_padding_mask.append(True)
            numels.append(numel_to_pad)
            total_numel += numel_to_pad
    self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(params_to_flatten, aligned_numel=0, requires_grad=flat_param_requires_grad)
    FlatParameter._init_metadata(self.flat_param, param_infos, numels, shapes, fqns, shared_param_infos, param_extensions, _convert_to_params(params_to_flatten) if use_orig_params else None, _convert_to_params(shared_params) if use_orig_params else None, is_padding_mask)