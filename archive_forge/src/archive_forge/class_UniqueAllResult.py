from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array