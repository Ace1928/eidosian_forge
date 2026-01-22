from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _wrap_acc(a: AggType, has_data: bool) -> WrappedAggType:
    """
    Wrap accumulation with a numeric boolean flag indicating whether or not
    this accumulation contains real data; if it doesn't, we consider it to be
    empty.

    Args:
        a: The accumulation value.
        has_data: Whether the accumulation contains real data.

    Returns:
        An AggType list with the last element being a numeric boolean flag indicating
        whether or not this accumulation contains real data. If the input a has length
        n, the returned AggType has length n + 1.
    """
    if not isinstance(a, list):
        a = [a]
    return a + [1 if has_data else 0]