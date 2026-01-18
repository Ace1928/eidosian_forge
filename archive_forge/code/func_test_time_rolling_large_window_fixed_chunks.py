from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('window,N', [('1s', 10), ('2s', 10), ('10s', 10), ('10h', 10), ('10s', 100), ('10h', 100)])
def test_time_rolling_large_window_fixed_chunks(window, N):
    df = pd.DataFrame({'a': pd.date_range('2016-01-01 00:00:00', periods=N, freq='1s'), 'b': np.random.randint(100, size=(N,))})
    df = df.set_index('a')
    ddf = dd.from_pandas(df, 5)
    assert_eq(ddf.rolling(window).sum(), df.rolling(window).sum())
    assert_eq(ddf.rolling(window).count(), df.rolling(window).count())
    assert_eq(ddf.rolling(window).mean(), df.rolling(window).mean())