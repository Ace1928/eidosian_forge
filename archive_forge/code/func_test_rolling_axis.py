from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='deprecated in pandas')
@pytest.mark.parametrize('kwargs', [dict(axis=0), dict(axis=1), dict(min_periods=1, axis=1), dict(axis='columns'), dict(axis='rows'), dict(axis='series')])
def test_rolling_axis(kwargs):
    df = pd.DataFrame(np.random.randn(20, 16))
    ddf = dd.from_pandas(df, npartitions=3)
    axis_deprecated_pandas = contextlib.nullcontext()
    if PANDAS_GE_210:
        axis_deprecated_pandas = pytest.warns(FutureWarning, match="'axis' keyword|Support for axis")
    axis_deprecated_dask = pytest.warns(FutureWarning, match="'axis' keyword is deprecated")
    if kwargs['axis'] == 'series':
        with axis_deprecated_pandas:
            expected = df[3].rolling(5, axis=0).std()
        with axis_deprecated_dask:
            result = ddf[3].rolling(5, axis=0).std()
        assert_eq(expected, result)
    else:
        with axis_deprecated_pandas:
            expected = df.rolling(3, **kwargs).mean()
        if kwargs['axis'] in (1, 'rows') and (not PANDAS_GE_210):
            ctx = pytest.warns(FutureWarning, match='Using axis=1 in Rolling')
        elif 'axis' in kwargs:
            ctx = axis_deprecated_dask
        with ctx:
            result = ddf.rolling(3, **kwargs).mean()
        assert_eq(expected, result)