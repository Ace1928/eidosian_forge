from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
def unique_values(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    res = np.unique(x._array, return_counts=False, return_index=False, return_inverse=False, equal_nan=False)
    return Array._new(res)