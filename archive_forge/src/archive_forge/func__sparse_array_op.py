from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
def _sparse_array_op(left: SparseArray, right: SparseArray, op: Callable, name: str) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith('__'):
        name = name[2:-2]
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype
    if ltype != rtype:
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)
        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype
    result_dtype = None
    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all='ignore'):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))
        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all='ignore'):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == 'r':
            left, right = (right, left)
            name = name[1:]
        if name in ('and', 'or', 'xor') and dtype == 'bool':
            opname = f'sparse_{name}_uint8'
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f'sparse_{name}_{dtype}'
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values
        if name in ['floordiv', 'mod'] and (right == 0).any() and (left.dtype.kind in 'iu'):
            opname = f'sparse_{name}_float64'
            left_sp_values = left_sp_values.astype('float64')
            right_sp_values = right_sp_values.astype('float64')
        sparse_op = getattr(splib, opname)
        with np.errstate(all='ignore'):
            result, index, fill = sparse_op(left_sp_values, left.sp_index, left.fill_value, right_sp_values, right.sp_index, right.fill_value)
    if name == 'divmod':
        return (_wrap_result(name, result[0], index, fill[0], dtype=result_dtype), _wrap_result(name, result[1], index, fill[1], dtype=result_dtype))
    if result_dtype is None:
        result_dtype = result.dtype
    return _wrap_result(name, result, index, fill, dtype=result_dtype)