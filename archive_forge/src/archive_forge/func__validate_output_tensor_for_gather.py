import collections.abc
import copy
from typing import Optional, List, Sequence
import torch
from torch.distributed import distributed_c10d
from torch.distributed import rpc
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.metadata import ShardMetadata
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
def _validate_output_tensor_for_gather(my_rank: int, dst_rank: int, size: torch.Size, dst_tensor: Optional[torch.Tensor]) -> None:
    if dst_rank == my_rank:
        if dst_tensor is None:
            raise ValueError(f'Argument ``dst_tensor`` must be specified on destination rank {dst_rank}')
        if tuple(size) != dst_tensor.size():
            raise ValueError(f'Argument ``dst_tensor`` have size {tuple(dst_tensor.size())},but should be {tuple(size)}')
    elif dst_tensor:
        raise ValueError('Argument ``dst_tensor`` must NOT be specified on non-destination ranks.')