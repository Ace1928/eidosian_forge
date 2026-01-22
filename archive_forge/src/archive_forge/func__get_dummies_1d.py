from __future__ import annotations
from collections import defaultdict
from collections.abc import (
import itertools
from typing import (
import numpy as np
from pandas._libs.sparse import IntIndex
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.series import Series
def _get_dummies_1d(data, prefix, prefix_sep: str | Iterable[str] | dict[str, str]='_', dummy_na: bool=False, sparse: bool=False, drop_first: bool=False, dtype: NpDtype | None=None) -> DataFrame:
    from pandas.core.reshape.concat import concat
    codes, levels = factorize_from_iterable(Series(data, copy=False))
    if dtype is None and hasattr(data, 'dtype'):
        input_dtype = data.dtype
        if isinstance(input_dtype, CategoricalDtype):
            input_dtype = input_dtype.categories.dtype
        if isinstance(input_dtype, ArrowDtype):
            import pyarrow as pa
            dtype = ArrowDtype(pa.bool_())
        elif isinstance(input_dtype, StringDtype) and input_dtype.storage != 'pyarrow_numpy':
            dtype = pandas_dtype('boolean')
        else:
            dtype = np.dtype(bool)
    elif dtype is None:
        dtype = np.dtype(bool)
    _dtype = pandas_dtype(dtype)
    if is_object_dtype(_dtype):
        raise ValueError('dtype=object is not a valid dtype for get_dummies')

    def get_empty_frame(data) -> DataFrame:
        index: Index | np.ndarray
        if isinstance(data, Series):
            index = data.index
        else:
            index = default_index(len(data))
        return DataFrame(index=index)
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)
    codes = codes.copy()
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = levels.insert(len(levels), np.nan)
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)
    number_of_cols = len(levels)
    if prefix is None:
        dummy_cols = levels
    else:
        dummy_cols = Index([f'{prefix}{prefix_sep}{level}' for level in levels])
    index: Index | None
    if isinstance(data, Series):
        index = data.index
    else:
        index = None
    if sparse:
        fill_value: bool | float
        if is_integer_dtype(dtype):
            fill_value = 0
        elif dtype == np.dtype(bool):
            fill_value = False
        else:
            fill_value = 0.0
        sparse_series = []
        N = len(data)
        sp_indices: list[list] = [[] for _ in range(len(dummy_cols))]
        mask = codes != -1
        codes = codes[mask]
        n_idx = np.arange(N)[mask]
        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)
        if drop_first:
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for col, ixs in zip(dummy_cols, sp_indices):
            sarr = SparseArray(np.ones(len(ixs), dtype=dtype), sparse_index=IntIndex(N, ixs), fill_value=fill_value, dtype=dtype)
            sparse_series.append(Series(data=sarr, index=index, name=col, copy=False))
        return concat(sparse_series, axis=1, copy=False)
    else:
        shape = (len(codes), number_of_cols)
        dummy_dtype: NpDtype
        if isinstance(_dtype, np.dtype):
            dummy_dtype = _dtype
        else:
            dummy_dtype = np.bool_
        dummy_mat = np.zeros(shape=shape, dtype=dummy_dtype, order='F')
        dummy_mat[np.arange(len(codes)), codes] = 1
        if not dummy_na:
            dummy_mat[codes == -1] = 0
        if drop_first:
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols, dtype=_dtype)