import collections
import itertools
import os
import warnings
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp.api import (
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import RemovableHandle
@no_type_check
def _init_param_handle_from_module(state: _FSDPState, fully_sharded_module: nn.Module, device_id: Optional[Union[int, torch.device]], param_init_fn: Optional[Callable[[nn.Module], None]], sync_module_states: bool) -> _FSDPState:
    """Initialize a ``FlatParamHandle`` from a module ``fully_sharded_module``."""
    _check_single_device_module(fully_sharded_module, state._ignored_params, device_id)
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(fully_sharded_module, state._ignored_params, state._ignored_modules)
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _materialize_with_param_init_fn(fully_sharded_module, param_init_fn, state._ignored_modules)
    elif is_meta_module:
        _materialize_meta_module(fully_sharded_module, device_id, state._ignored_modules)
    elif is_torchdistX_deferred_init:
        deferred_init.materialize_module(fully_sharded_module, check_fn=lambda submodule: _get_module_fsdp_state(submodule) is None and submodule not in state._ignored_modules)
    ignored_buffers = {buffer for ignored_module in state._ignored_modules for buffer in ignored_module.buffers()}
    _move_module_to_device(fully_sharded_module, state._ignored_params, ignored_buffers, device_from_device_id)
    state.compute_device = _get_compute_device(fully_sharded_module, state._ignored_params, device_from_device_id, state.rank)
    managed_params = list(_get_orig_params(fully_sharded_module, state._ignored_params))
    if sync_module_states:
        _sync_module_params_and_buffers(fully_sharded_module, managed_params, state.process_group)
        if state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            _sync_module_params_and_buffers(fully_sharded_module, managed_params, state._inter_node_pg)
    _init_param_handle_from_params(state, managed_params, fully_sharded_module)
    return state