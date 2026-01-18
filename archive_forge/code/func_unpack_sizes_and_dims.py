import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.op_schema import OpSchema
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def unpack_sizes_and_dims(sizes: List[Union[DSymInt, int]], mesh: DeviceMesh) -> Tuple[List[int], List[Placement]]:
    local_sizes: List[int] = [s.local_value if isinstance(s, DSymInt) else s for s in sizes]
    placements: List[Placement] = [Shard(i) for i, a in enumerate(sizes) if isinstance(a, DSymInt) and a.is_shard()] or [Replicate()]
    assert len(placements) == mesh.ndim, f'The number of sharded dimensions ({len(placements)}) must match number of dimensions in device mesh ({mesh.ndim}).'
    return (local_sizes, placements)