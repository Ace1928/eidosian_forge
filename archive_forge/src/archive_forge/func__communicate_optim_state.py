import copy
import functools
import logging
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp._fsdp_extensions import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.utils._pytree import tree_map_only
def _communicate_optim_state(fsdp_param_info: FSDPParamInfo, flat_param_state: Dict[str, Any]) -> _ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flat parameter across ranks. All
    ranks will hold the entire non-sharded optimizer state on GPU.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flat parameter.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for the target
        flat parameter.
    """
    fsdp_state = fsdp_param_info.state
    flat_param = fsdp_param_info.handle.flat_param
    state = _ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = (state.tensor_state, state.zero_dim_tensor_state, state.non_tensor_state)
    for state_name, value in sorted_items(flat_param_state):
        if torch.is_tensor(value) and value.dim() > 0:
            if fsdp_state.world_size == 1 or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
                tensor_state[state_name] = value
                continue
            assert fsdp_state.compute_device is not None, 'compute_device has not been initialized'
            if value.device.type != fsdp_state.compute_device.type:
                value = value.to(fsdp_state.compute_device)
            buffer_size = flat_param._full_param_padded.size()
            tensor_buffer = value.new_zeros(*buffer_size)
            dist.all_gather_into_tensor(tensor_buffer, value, group=fsdp_state.process_group)
            fsdp_state._device_handle.synchronize()
            unpadded_numel = cast(nn.Parameter, flat_param._unpadded_unsharded_size).numel()
            tensor_state[state_name] = tensor_buffer[:unpadded_numel]
        elif _is_zero_dim_tensor(value):
            zero_dim_tensor_state[state_name] = value.detach().clone()
        else:
            non_tensor_state[state_name] = value
    return state