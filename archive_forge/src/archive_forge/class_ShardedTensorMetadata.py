from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch
from torch.distributed._shard.metadata import ShardMetadata
@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """
    shards_metadata: List[ShardMetadata] = field(default_factory=list)
    size: torch.Size = field(default=torch.Size([]))
    tensor_properties: TensorProperties = field(default_factory=TensorProperties)