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
def _validate_ragged_properties(start_indices, flat_array):
    """
    Validate that start_indices are flat_array arrays that may be used to
    represent a valid RaggedArray.

    Parameters
    ----------
    flat_array: numpy array containing concatenation
                of all nested arrays to be represented
                by this ragged array
    start_indices: unsigned integer numpy array the same
                   length as the ragged array where values
                   represent the index into flat_array where
                   the corresponding ragged array element
                   begins
    Raises
    ------
    ValueError:
        if input arguments are invalid or incompatible properties
    """
    if not isinstance(start_indices, np.ndarray) or start_indices.dtype.kind != 'u' or start_indices.ndim != 1:
        raise ValueError("\nThe start_indices property of a RaggedArray must be a 1D numpy array of\nunsigned integers (start_indices.dtype.kind == 'u')\n    Received value of type {typ}: {v}".format(typ=type(start_indices), v=repr(start_indices)))
    if not isinstance(flat_array, np.ndarray) or flat_array.ndim != 1:
        raise ValueError('\nThe flat_array property of a RaggedArray must be a 1D numpy array\n    Received value of type {typ}: {v}'.format(typ=type(flat_array), v=repr(flat_array)))
    invalid_inds = start_indices > len(flat_array)
    if invalid_inds.any():
        some_invalid_vals = start_indices[invalid_inds[:10]]
        raise ValueError('\nElements of start_indices must be less than the length of flat_array ({m})\n    Invalid values include: {vals}'.format(m=len(flat_array), vals=repr(some_invalid_vals)))