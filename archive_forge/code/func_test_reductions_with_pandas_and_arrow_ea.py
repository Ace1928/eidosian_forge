from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.parametrize('dtype', [pytest.param('int64[pyarrow]', marks=pytest.mark.skipif(pa is None or not PANDAS_GE_150, reason='requires pyarrow installed')), pytest.param('float64[pyarrow]', marks=pytest.mark.skipif(pa is None or not PANDAS_GE_150, reason='requires pyarrow installed')), 'Int64', 'Int32', 'Float64', 'UInt64'])
@pytest.mark.parametrize('func', ['std', 'var', 'skew', 'kurtosis'])
def test_reductions_with_pandas_and_arrow_ea(dtype, func):
    if func in ['skew', 'kurtosis']:
        pytest.importorskip('scipy')
        if 'pyarrow' in dtype:
            pytest.xfail('skew/kurtosis not implemented for arrow dtypes')
    ser = pd.Series([1, 2, 3, 4], dtype=dtype)
    ds = dd.from_pandas(ser, npartitions=2)
    pd_result = getattr(ser, func)()
    dd_result = getattr(ds, func)()
    if func == 'kurtosis':
        n = ser.shape[0]
        factor = (n - 1) * (n + 1) / ((n - 2) * (n - 3))
        offset = 6 * (n - 1) / ((n - 2) * (n - 3))
        dd_result = factor * dd_result + offset
    assert_eq(dd_result, pd_result, check_dtype=False)