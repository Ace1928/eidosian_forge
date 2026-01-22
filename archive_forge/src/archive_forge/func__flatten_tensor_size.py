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
def _flatten_tensor_size(size) -> torch.Size:
    """
    Checks if tensor size is valid, then flatten/return a torch.Size object.
    """
    if len(size) == 1 and isinstance(size[0], collections.abc.Sequence):
        dims = list(*size)
    else:
        dims = list(size)
    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f'size has to be a sequence of ints, found: {dims}')
    return torch.Size(dims)