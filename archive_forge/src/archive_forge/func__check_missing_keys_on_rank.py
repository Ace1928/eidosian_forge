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
def _check_missing_keys_on_rank(r0_optim_state_keys: List[_OptimStateKey], optim_state_key_to_param_key: Dict[_OptimStateKey, Union[str, int]], param_key_to_param: Dict[Union[str, int], nn.Parameter], group: Optional[dist.ProcessGroup]) -> None:
    missing_keys: List[_OptimStateKey] = []
    for r0_optim_state_key in r0_optim_state_keys:
        if r0_optim_state_key not in optim_state_key_to_param_key:
            missing_keys.append(r0_optim_state_key)
            continue
        param_key = optim_state_key_to_param_key[r0_optim_state_key]
        if isinstance(param_key, int):
            assert param_key >= 0 and param_key < len(param_key_to_param), 'Check the `param_key_to_param` construction'
    device = _get_pg_default_device(group)
    num_missing = torch.tensor([len(missing_keys)], dtype=torch.int32, device=device)
    dist.all_reduce(num_missing, group=group)
    if num_missing.item() > 0:
        obj_list = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(obj_list, missing_keys, group=group)
        error_msg = "FSDP currently requires each rank to have at least the optimizer states needed by rank 0's optimizer but some ranks are missing some of those states"
        for rank, keys in enumerate(obj_list):
            keys = cast(List[_OptimStateKey], keys)
            if len(keys) > 0:
                error_msg += f'\nRank {rank} is missing states for the parameters: {[key.unflat_param_names for key in keys]}'
        raise RuntimeError(error_msg)