from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch
from torch.distributed._shard.metadata import ShardMetadata
class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2