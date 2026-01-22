from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def _next_power_of_2_or_max(n: int, max_n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n, with a limit.

    Useful when used in splitting a tensor into chunks with power-of-2 sizes.
    """
    if n == 0:
        return 1
    orig_n = n
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    assert n >= orig_n, f'{n} vs. {orig_n}'
    assert bin(n).count('1') == 1, bin(n)
    if n > max_n:
        return max_n
    return n