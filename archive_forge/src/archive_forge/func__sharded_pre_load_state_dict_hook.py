import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple
import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _replace_by_prefix
from ._fsdp_extensions import (
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM
@no_type_check
def _sharded_pre_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    _lazy_init(fsdp_state, module)
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f'{FSDP_PREFIX}')
    if not _has_fsdp_params(fsdp_state, module):
        return
    handle = _module_handle(fsdp_state, module)
    if not handle.uses_sharded_strategy:
        raise RuntimeError('load_sharded_state_dict can only be called when parameters are flattened and sharded.')
    fqn_to_param_ext = dict(zip(handle.flat_param._fqns, handle.flat_param._param_extensions))
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if not _is_composable(fsdp_state):
            fqn_from_global_root = f'{prefix}{FSDP_PREFIX}{fqn}'
        else:
            fqn_from_global_root = f'{prefix}{fqn}'
        try:
            param = state_dict.pop(fqn_from_global_root)
        except KeyError:
            logger.warning(f'Did not find param with FQN {fqn_from_global_root}, skipping it. The weight will not be filled if you expect it to be.')
            continue
        if not fsdp_state._state_dict_config._use_dtensor:
            param, shards = _ext_pre_load_state_dict_transform(param, fsdp_state._fsdp_extension)
            assert len(shards) < 2, f'Expects 0 or 1 shard per rank but got {len(shards)} shards on rank {fsdp_state.rank}.'
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = math.ceil(dim_0_size / fsdp_state.world_size) * param_numel // dim_0_size
            if len(shards) == 1:
                local_tensor = shards[0].tensor.flatten()
                with SimpleProfiler.profile(SimpleProfiler.Type.H2D):
                    local_tensor = local_tensor.to(fsdp_state.compute_device)
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(chunk_size, dtype=param.dtype, device=fsdp_state.compute_device)
            tensor = torch.empty(chunk_size * fsdp_state.world_size, dtype=local_tensor.dtype, device=fsdp_state.compute_device)
            with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER):
                dist.all_gather_into_tensor(tensor, local_tensor, group=fsdp_state.process_group)
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            state_dict[fqn_from_global_root] = tensor
        else:
            if param.device != fsdp_state._device_mesh.device_type:
                param = param.to(fsdp_state._device_mesh.device_type)
            parent_mesh = _mesh_resources.get_parent_mesh(fsdp_state._device_mesh)
            local_tensor = _ext_all_gather_dtensor(param, parent_mesh, fsdp_state._fsdp_extension)
            if fqn_to_param_ext.get(fqn) is not None:
                ext = fqn_to_param_ext[fqn]
                local_tensor = _ext_post_unflatten_transform(local_tensor, ext, fsdp_state._fsdp_extension)
            state_dict[fqn_from_global_root] = local_tensor
    with SimpleProfiler.profile('_enter_unshard_params_ctx'):
        _enter_unshard_params_ctx(module, fsdp_state, writeback=True)