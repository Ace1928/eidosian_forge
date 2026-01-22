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
def _make_sparse(arr: np.ndarray, kind: SparseIndexKind='block', fill_value=None, dtype: np.dtype | None=None):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    assert isinstance(arr, np.ndarray)
    if arr.ndim > 1:
        raise TypeError('expected dimension <= 1 data')
    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)
    if isna(fill_value):
        mask = notna(arr)
    else:
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)
        if is_object_dtype(arr.dtype):
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value
    length = len(arr)
    if length != len(mask):
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)
    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)
    return (sparsified_values, index, fill_value)