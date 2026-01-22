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
def _convert_state_with_flat_params(all_optim_state_keys: List[_OptimStateKey], optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]], fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo], optim_state_dict: Dict[Union[str, int], Any], to_save: bool, shard_state: bool, cpu_offload: bool=True) -> Dict[str, Any]:
    fsdp_osd_state: Dict[str, Any] = {}
    for optim_state_key in all_optim_state_keys:
        param_key: Union[str, int, None] = optim_state_key_to_param_key.get(optim_state_key, None)
        assert param_key is not None, f'If use_orig_params is False, we must be able to find the corresponding param id. {optim_state_key} {param_key}'
        if optim_state_key.is_fsdp_managed:
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            unflat_state = _unflatten_optim_state(fsdp_param_info, optim_state_dict[param_key], to_save, shard_state, cpu_offload)
            if to_save:
                assert len(unflat_state) == len(optim_state_key.unflat_param_names)
                for unflat_param_name, unflat_param_state in zip(optim_state_key.unflat_param_names, unflat_state):
                    fsdp_osd_state[unflat_param_name] = unflat_param_state
        elif to_save:
            assert len(optim_state_key.unflat_param_names) == 1
            unflat_param_name = optim_state_key.unflat_param_names[0]
            fsdp_osd_state[unflat_param_name] = copy.copy(optim_state_dict[param_key])
            if cpu_offload:
                for state_name, value in sorted_items(fsdp_osd_state[unflat_param_name]):
                    if not torch.is_tensor(value):
                        continue
                    fsdp_osd_state[unflat_param_name][state_name] = value.cpu()
    return fsdp_osd_state