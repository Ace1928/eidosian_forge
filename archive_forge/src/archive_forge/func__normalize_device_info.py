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
def _normalize_device_info(device_type: str, device_id: int) -> str:
    """Device info normalization."""
    if device_type == 'cpu':
        return 'cpu'
    return f'{device_type}:{device_id}'