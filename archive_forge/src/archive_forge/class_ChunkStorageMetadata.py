from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Sequence, Any
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.checkpoint.stateful import StatefulT
import torch
from torch.distributed._shard.sharded_tensor import (
@dataclass
class ChunkStorageMetadata:
    """Each chunk is expected to have the same properties of the TensorStorageMetadata that includes it."""
    offsets: torch.Size
    sizes: torch.Size