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
def _broadcast_processed_state(fsdp_state: _FSDPState, optim_state: Dict[str, Any], group: Optional[dist.ProcessGroup]) -> Dict[str, Any]:
    objects: List[Any] = [None]
    if fsdp_state.rank == 0:
        objects[0] = tree_map_only(torch.Tensor, lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype), optim_state)
    dist.broadcast_object_list(objects, src=0, group=group)
    if fsdp_state.rank == 0:
        return optim_state
    else:
        return objects[0]