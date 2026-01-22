from enum import Enum
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence
import torch
import torch.distributed as dist
import torch.nn.functional as F
class ProcessGroupName(str, Enum):
    default = 'default'
    reduce_scatter = 'reduce_scatter'