from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def _infer_dtype(array, name=None):
    """Given an object array with no missing values, infer its dtype from all elements."""
    if array.dtype.kind != 'O':
        raise TypeError('infer_type must be called on a dtype=object array')
    if array.size == 0:
        return np.dtype(float)
    native_dtypes = set(np.vectorize(type, otypes=[object])(array.ravel()))
    if len(native_dtypes) > 1 and native_dtypes != {bytes, str}:
        raise ValueError('unable to infer dtype on variable {!r}; object array contains mixed native types: {}'.format(name, ', '.join((x.__name__ for x in native_dtypes))))
    element = array[(0,) * array.ndim]
    if isinstance(element, bytes):
        return strings.create_vlen_dtype(bytes)
    elif isinstance(element, str):
        return strings.create_vlen_dtype(str)
    dtype = np.array(element).dtype
    if dtype.kind != 'O':
        return dtype
    raise ValueError(f'unable to infer dtype on variable {name!r}; xarray cannot serialize arbitrary Python objects')