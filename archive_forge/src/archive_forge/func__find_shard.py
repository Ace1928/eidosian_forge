import os
import io
import itertools
from typing import (
import torch.distributed as dist
from .api import (
import torch
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DTensor
from .metadata import (
def _find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    if index.offset is None:
        raise ValueError(f'Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided')
    shards = tensor.local_shards()
    if index.index is not None:
        if len(shards) > index.index and torch.Size(shards[index.index].metadata.shard_offsets) == index.offset:
            return shards[index.index]
    for shard in shards:
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard
    raise ValueError(f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'")