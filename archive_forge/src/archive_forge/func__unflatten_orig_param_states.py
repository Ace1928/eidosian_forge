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
def _unflatten_orig_param_states(fsdp_param_info: FSDPParamInfo, output_states: Dict[str, Dict[str, Any]], state_name: str, shard_state: bool, to_save: bool, cpu_offload: bool) -> None:
    """
    Given a output state dict, ``output_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parmeter ID), and the values
    are gathered states, unflatten the states to the original dimensions.

    This function performs the unflattening process in-place.
    """
    if not to_save:
        return
    flat_param = fsdp_param_info.handle.flat_param
    fsdp_state = fsdp_param_info.state
    for fqn, gathered_state in output_states.items():
        value = gathered_state[state_name]
        param_idx = fsdp_param_info.param_indices[fqn]
        if isinstance(value, DTensor):
            placement = value.placements[0]
            if placement != Replicate():
                placement_dim = placement.dim
                value_local = value.redistribute(placements=(Replicate(),))
                reshape_size = list(flat_param._shapes[param_idx])
                reshape_size[placement_dim] *= 2
                reshape_size = torch.Size(reshape_size)
                value = value.reshape(reshape_size)
            else:
                value = value.reshape(flat_param._shapes[param_idx])
        else:
            value = value.reshape(flat_param._shapes[param_idx])
        if shard_state:
            osd_config = fsdp_state._optim_state_dict_config
            if getattr(osd_config, '_use_dtensor', False):
                assert fsdp_state._device_mesh is not None
                value = _ext_chunk_dtensor(value, fsdp_state.rank, fsdp_state._device_mesh, fsdp_state._fsdp_extension)
            else:
                assert fsdp_state.process_group is not None
                value = _ext_chunk_tensor(value, fsdp_state.rank, fsdp_state.world_size, fsdp_state._device_handle.device_count(), fsdp_state.process_group, fsdp_state._fsdp_extension)
        elif not cpu_offload:
            with SimpleProfiler.profile('clone'):
                value = value.detach().clone()
        if cpu_offload:
            with SimpleProfiler.profile(SimpleProfiler.Type.D2H):
                value = value.cpu()
        gathered_state[state_name] = value