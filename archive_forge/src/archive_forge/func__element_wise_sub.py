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
def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a - i_b for i_a, i_b in zip(a, b)]