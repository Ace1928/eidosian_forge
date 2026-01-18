from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(PANDAS_GE_200 and (not PANDAS_GE_210), reason='buggy pandas implementation')
def test_rolling_numba_engine():
    pytest.importorskip('numba')
    df = pd.DataFrame({'A': range(5), 'B': range(0, 10, 2)})
    ddf = dd.from_pandas(df, npartitions=3)

    def f(x):
        return np.sum(x) + 5
    assert_eq(df.rolling(3).apply(f, engine='numba', raw=True), ddf.rolling(3).apply(f, engine='numba', raw=True))