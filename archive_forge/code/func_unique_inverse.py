from __future__ import annotations
from ._array_object import Array
from typing import NamedTuple
import cupy as np
def unique_inverse(x: Array, /) -> UniqueInverseResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, inverse_indices = np.unique(x._array, return_counts=False, return_index=False, return_inverse=True, equal_nan=False)
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(Array._new(values), Array._new(inverse_indices))