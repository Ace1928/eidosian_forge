from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_series_resample_non_existent_datetime():
    index = [pd.Timestamp('2016-10-15 00:00:00'), pd.Timestamp('2016-10-16 10:00:00'), pd.Timestamp('2016-10-17 00:00:00')]
    df = pd.DataFrame([[1], [2], [3]], index=index)
    df.index = df.index.tz_localize('America/Sao_Paulo')
    ddf = dd.from_pandas(df, npartitions=1)
    result = ddf.resample('1D').mean()
    expected = df.resample('1D').mean()
    assert_eq(result, expected, check_freq=False)