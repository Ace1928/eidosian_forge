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
@torch.no_grad()
def _post_load_state_dict_hook(module: nn.Module, *args: Any) -> None:
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn('When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict willbe returned.')
    else:
        context = contextlib.nullcontext()
    with context:
        _post_load_state_dict_hook_fn = {StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook, StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook, StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook}
        _post_load_state_dict_hook_fn[fsdp_state._state_dict_type](module, fsdp_state)
    if fsdp_state._is_root:
        SimpleProfiler.dump_and_reset('FSDP model load_state_dict profiling: ')