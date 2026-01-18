from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
def unique_counts(x: Array, /) -> UniqueCountsResult:
    res = np.unique(x._array, return_counts=True, return_index=False, return_inverse=False, equal_nan=False)
    return UniqueCountsResult(*[Array._new(i) for i in res])