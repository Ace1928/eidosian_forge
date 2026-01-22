from typing import Any, List
import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._tensor import DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
from .metadata import (
from .planner import (
from .resharding import (
def _create_read_item_for_tensor(dest_index, dest_offsets, storage_index, storage_offsets, lengths):
    return ReadItem(type=LoadItemType.TENSOR, dest_index=dest_index, dest_offsets=torch.Size(dest_offsets), storage_index=storage_index, storage_offsets=torch.Size(storage_offsets), lengths=torch.Size(lengths))