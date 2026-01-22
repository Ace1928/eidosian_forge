import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase
def _rand_maxed_partition(r: random.Random, total: int, n: int, mx: int, positive: bool=True) -> List[int]:
    if positive:
        total -= n
        mx -= 1
    idxs = r.sample(range(n * mx), total)
    y = torch.zeros(n, mx, dtype=torch.int32)
    y.flatten()[idxs] = 1
    z = y.sum(1)
    if positive:
        z += 1
    return z.tolist()