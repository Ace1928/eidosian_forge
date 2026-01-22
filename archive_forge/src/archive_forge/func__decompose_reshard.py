from typing import cast, Dict, List, Tuple
import torch
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def _decompose_reshard(val: List[_PlacementItem]) -> List[_PlacementItem]:
    """
    Decompose Si -> Sj into Si -> R -> Sj
    There's 2 ways a shardings can differ within a mesh dimension:
      1) sharding on different tensor dimensions, e.g. Shard(0) -> Shard(1)
      2) different sub-shards of a repeated shard ("mis-aligned sharding")
          (Shard(0), Shard(0)) -> (Replicate(), Shard(0))
          Here the Shard(0) -> Shard(0) for mesh dimension 2 is actually
          a reshard, because in the first case it's a sub-sharding of an already tensor dimension 0,
          and in the second case, it's the first sharding on tensor dimension 0.
    """
    from collections import defaultdict
    repeat_dim_current: Dict[int, int] = defaultdict(int)
    repeat_dim_target: Dict[int, int] = defaultdict(int)
    output: List[_PlacementItem] = []
    for i, (current, target) in val:
        if current.is_shard():
            repeat_dim_current[cast(Shard, current).dim] += 1
        if target.is_shard():
            repeat_dim_target[cast(Shard, target).dim] += 1
        if isinstance(current, Shard) and isinstance(target, Shard) and (current.dim != target.dim or repeat_dim_current[current.dim] != repeat_dim_target[target.dim]):
            output.append((i, (current, Replicate())))
            output.append((i, (Replicate(), target)))
        else:
            output.append((i, (current, target)))
    return output