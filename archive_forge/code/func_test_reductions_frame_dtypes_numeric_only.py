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
@pytest.mark.parametrize('func', ['mean', 'sem'])
def test_reductions_frame_dtypes_numeric_only(func):
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2011, i, 1) for i in range(1, 8)], 'str': list('abcdefgh'), 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, np.nan]), 'bool': [True, False] * 4})
    ddf = dd.from_pandas(df, 3)
    kwargs = {'numeric_only': True}
    assert_eq(getattr(df, func)(**kwargs), getattr(ddf, func)(**kwargs))
    if not DASK_EXPR_ENABLED:
        with pytest.raises(NotImplementedError, match="'numeric_only=False"):
            getattr(ddf, func)(numeric_only=False)
    assert_eq(df.sem(ddof=0, **kwargs), ddf.sem(ddof=0, **kwargs))
    assert_eq(df.std(ddof=0, **kwargs), ddf.std(ddof=0, **kwargs))
    assert_eq(df.var(ddof=0, **kwargs), ddf.var(ddof=0, **kwargs))
    assert_eq(df.var(skipna=False, **kwargs), ddf.var(skipna=False, **kwargs))
    assert_eq(df.var(skipna=False, ddof=0, **kwargs), ddf.var(skipna=False, ddof=0, **kwargs))
    if not DASK_EXPR_ENABLED:
        assert_eq(df._get_numeric_data(), ddf._get_numeric_data())
    df_numerics = df[['int', 'float', 'bool']]
    ddf_numerics = ddf[['int', 'float', 'bool']]
    if not DASK_EXPR_ENABLED:
        assert_eq(df_numerics, ddf._get_numeric_data())
        assert ddf_numerics._get_numeric_data().dask == ddf_numerics.dask
    assert_eq(getattr(df_numerics, func)(), getattr(ddf_numerics, func)())