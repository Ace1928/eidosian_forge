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
def _flatten_optim_state(fsdp_param_info: FSDPParamInfo, unflat_osd_state: Dict[str, Dict[str, Any]], unflat_param_names: List[str]) -> Dict[str, Any]:
    """
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flat parameter in ``fsdp_param_info`` corresponding to the unflattened
    parameter names in ``unflat_param_names``.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flat parameter ``flat_param``.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flat parameter. The sharded optimizer state dict's "state"
        part will map a key to this returned value.
    """
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param
    num_unflat_params = len(unflat_param_names)
    assert num_unflat_params > 0, 'Expects at least one unflattened parameter corresponding to the flat parameter'
    unflat_param_shapes = flat_param._shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    assert num_unflat_params == num_unflat_param_shapes, f'Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}'
    has_state = [bool(unflat_param_name in unflat_osd_state) for unflat_param_name in unflat_param_names]
    if not any(has_state):
        return {}
    unflat_param_states = [_gather_state_dict(unflat_osd_state[unflat_param_name], pg=fsdp_state.process_group, device=fsdp_state.compute_device) if unflat_param_name in unflat_osd_state else None for unflat_param_name in unflat_param_names]
    state_names = None
    for unflat_param_state in unflat_param_states:
        if unflat_param_state is None:
            continue
        if state_names is None:
            state_names = set(unflat_param_state.keys())
        elif state_names != set(unflat_param_state.keys()):
            raise ValueError(f'Differing optimizer state names for the unflattened parameters: {unflat_param_names}')
    assert state_names is not None
    flat_state: Dict[str, Any] = {}
    for state_name in state_names:
        state_values = [unflat_param_state[state_name] if unflat_param_state is not None else None for unflat_param_state in unflat_param_states]
        non_none_state_values = [v for v in state_values if v is not None]
        if not non_none_state_values:
            flat_state[state_name] = None
            continue
        are_pos_dim_tensors = are_zero_dim_tensors = are_non_tensors = True
        for v in non_none_state_values:
            are_pos_dim_tensors &= torch.is_tensor(v) and v.dim() > 0
            are_zero_dim_tensors &= _is_zero_dim_tensor(v)
            are_non_tensors &= not torch.is_tensor(v)
        types = {type(v) for v in non_none_state_values}
        if len(types) != 1 or not (are_pos_dim_tensors or are_zero_dim_tensors or are_non_tensors):
            raise ValueError(f'Differing optimizer state types for state {state_name}, values {non_none_state_values}, and unflattened parameter names {unflat_param_names}')
        if are_pos_dim_tensors:
            flat_tensor = _flatten_tensor_optim_state(state_name, state_values, unflat_param_names, unflat_param_shapes, handle)
            if fsdp_state.world_size != 1 and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD:
                sharded_flat_tensor, _ = FlatParamHandle._get_shard(flat_tensor, fsdp_state.rank, fsdp_state.world_size)
            else:
                sharded_flat_tensor = flat_tensor
            flat_state[state_name] = sharded_flat_tensor
        elif are_zero_dim_tensors:
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(state_name, state_values, unflat_param_names)
        else:
            assert are_non_tensors
            flat_state[state_name] = _flatten_non_tensor_optim_state(state_name, state_values, unflat_param_names)
    return flat_state