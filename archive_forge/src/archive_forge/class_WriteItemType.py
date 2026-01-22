import abc
from dataclasses import dataclass
import io
from typing import List, Tuple, Any, Union, Optional
from enum import Enum, auto
import torch
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from .metadata import (
class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()