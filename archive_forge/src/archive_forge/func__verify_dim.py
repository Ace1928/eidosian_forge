from dataclasses import dataclass
import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import (
from torch.distributed._shard._utils import narrow_tensor
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from typing import List, Union, TYPE_CHECKING
from ._internals import (
from .api import ShardingSpec
@staticmethod
def _verify_dim(dim):
    if isinstance(dim, str):
        raise NotImplementedError('ChunkShardingSpec does not support named dimension yet!')
    if not isinstance(dim, int):
        raise ValueError(f'Sharding dim needs to be an integer, found: {dim}')