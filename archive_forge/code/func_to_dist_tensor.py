import itertools
import sys
from functools import wraps
from typing import (
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
from torch.distributed._tensor import (
from torch.distributed._tensor.placement_types import Placement
def to_dist_tensor(self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]) -> torch.Tensor:
    if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
        if self.is_supported_tensor(t):
            self.hit += 1
            if t.ndim == 0:
                r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
            else:
                r = distribute_tensor(t, mesh, placements)
            if type(t) is torch.nn.Parameter:
                r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
            return r
        else:
            self.miss += 1
            return t
    elif torch.overrides.is_tensor_like(t):
        self.miss += 1
        return t
    else:
        raise RuntimeError(f'Trying to convert to DTensor, but got {type(t)}')