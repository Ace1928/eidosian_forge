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
def _all_gather_keys(local_dict: Dict[Any, Any]) -> List[Any]:
    """Gathers all keys, and returns them sorted."""
    keys = list(local_dict.keys())
    gathered_keys: List[List[Any]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_keys, keys)
    return sorted(set(itertools.chain.from_iterable(gathered_keys)))