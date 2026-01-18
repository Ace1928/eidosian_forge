import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def unravel_flat_index(flat_index: int) -> Tuple[int, ...]:
    if not matches.shape:
        return ()
    inverse_index = []
    for size in matches.shape[::-1]:
        div, mod = divmod(flat_index, size)
        flat_index = div
        inverse_index.append(mod)
    return tuple(inverse_index[::-1])