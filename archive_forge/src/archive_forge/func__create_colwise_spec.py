import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
from torch.distributed.checkpoint.planner import LoadPlan
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import (
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import (
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.remote_device import _remote_device
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.default_planner import (
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.utils import (
from torch._utils import _get_device_module
def _create_colwise_spec(pg: Optional[dist.ProcessGroup]=None) -> ChunkShardingSpec:
    pg_device_type = dist.distributed_c10d._get_pg_default_device(pg).type
    if pg is None:
        placements = [f'rank:{idx}/{_gen_rank_device(idx, pg_device_type)}' for idx in range(dist.get_world_size())]
    else:
        placements = [f'rank:{idx}/{_gen_rank_device(dist.get_global_rank(pg, idx), pg_device_type)}' for idx in range(pg.size())]
    return ChunkShardingSpec(dim=0, placements=cast(List[Union[_remote_device, str]], placements))