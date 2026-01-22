from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@jit(nopython=True, nogil=True)
def _eq_ragged_scalar(start_indices, flat_array, val):
    """
    Compare elements of a RaggedArray with a scalar array

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    val: ndarray

    Returns
    -------
    mask: ndarray
        1D bool array of same length as inputs with elements True when
        ragged element equals scalar val, False otherwise.
    """
    n = len(start_indices)
    m = len(flat_array)
    cols = len(val)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start_index = start_indices[i]
        stop_index = start_indices[i + 1] if i < n - 1 else m
        if stop_index - start_index != cols:
            el_equal = False
        else:
            el_equal = True
            for val_index, flat_index in enumerate(range(start_index, stop_index)):
                el_equal &= flat_array[flat_index] == val[val_index]
        result[i] = el_equal
    return result