from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
@meta_nonempty.register(pd.Series)
def _nonempty_series(s, idx=None):
    if idx is None:
        idx = _nonempty_index(s.index)
    dtype = s.dtype
    if len(s) > 0:
        data = [s.iloc[0]] * 2
    elif isinstance(dtype, pd.DatetimeTZDtype):
        entry = pd.Timestamp('1970-01-01', tz=dtype.tz)
        data = [entry, entry]
    elif isinstance(dtype, pd.CategoricalDtype):
        if len(s.cat.categories):
            data = [s.cat.categories[0]] * 2
            cats = s.cat.categories
        else:
            data = _nonempty_index(s.cat.categories)
            cats = s.cat.categories[:0]
        data = pd.Categorical(data, categories=cats, ordered=s.cat.ordered)
    elif is_integer_na_dtype(dtype):
        data = pd.array([1, None], dtype=dtype)
    elif is_float_na_dtype(dtype):
        data = pd.array([1.0, None], dtype=dtype)
    elif isinstance(dtype, pd.PeriodDtype):
        freq = dtype.freq
        data = [pd.Period('2000', freq), pd.Period('2001', freq)]
    elif isinstance(dtype, pd.SparseDtype):
        entry = _scalar_from_dtype(dtype.subtype)
        data = pd.array([entry, entry], dtype=dtype)
    elif isinstance(dtype, pd.IntervalDtype):
        entry = _scalar_from_dtype(dtype.subtype)
        data = pd.array([entry, entry], dtype=dtype)
    elif type(dtype) in make_array_nonempty._lookup:
        data = make_array_nonempty(dtype)
    else:
        entry = _scalar_from_dtype(dtype)
        data = np.array([entry, entry], dtype=dtype)
    out = pd.Series(data, name=s.name, index=idx)
    out.attrs = s.attrs
    return out