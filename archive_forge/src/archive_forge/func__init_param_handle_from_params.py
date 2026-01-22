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
def _init_param_handle_from_params(state: _FSDPState, params: List[nn.Parameter], fully_sharded_module: nn.Module):
    if len(params) == 0:
        return
    handle = FlatParamHandle(params, fully_sharded_module, state.compute_device, SHARDING_STRATEGY_MAP[state.sharding_strategy], state.cpu_offload.offload_params, state.mixed_precision.param_dtype, state.mixed_precision.reduce_dtype, state.mixed_precision.keep_low_precision_grads, state.process_group, state._use_orig_params, fsdp_extension=state._fsdp_extension)
    handle.shard()
    assert not state._handle
    state.params.append(handle.flat_param)
    state._handle = handle
    state._fully_sharded_module_to_handle[handle._fully_sharded_module] = handle
    cpu_device = torch.device('cpu')
    if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
        handle.flat_param_to(cpu_device)