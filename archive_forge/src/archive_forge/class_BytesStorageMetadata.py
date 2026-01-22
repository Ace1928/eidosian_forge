from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Sequence, Any
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.checkpoint.stateful import StatefulT
import torch
from torch.distributed._shard.sharded_tensor import (
@dataclass
class BytesStorageMetadata:
    pass