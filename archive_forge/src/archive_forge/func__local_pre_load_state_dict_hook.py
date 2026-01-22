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
def _local_pre_load_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    _lazy_init(fsdp_state, module)
    _replace_by_prefix(state_dict, prefix, f'{prefix}{FSDP_PREFIX}')
    fqn = f'{prefix}{FSDP_PREFIX}{FLAT_PARAM}'
    if fqn not in state_dict:
        assert not _has_fsdp_params(fsdp_state, module), 'No `FlatParameter` in `state_dict` for this FSDP instance but it has parameters'
        return
    load_tensor = state_dict[fqn]
    assert isinstance(load_tensor, ShardedTensor), 'Tensors in local_state_dict should be ShardedTensor.'
    flat_param = _module_handle(fsdp_state, module).flat_param
    assert flat_param is not None
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    shards = load_tensor.local_shards()
    if valid_data_size > 0:
        assert len(shards), 'load_local_state_dict assume one shard per ShardedTensor.'
        load_tensor = shards[0].tensor
        if flat_param._shard_numel_padded > 0:
            assert load_tensor.numel() < flat_param.numel(), f'Local shard size = {flat_param.numel()} and the tensor in the state_dict is {load_tensor.numel()}.'
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    else:
        load_tensor = flat_param
    state_dict[fqn] = load_tensor