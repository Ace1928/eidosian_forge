from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_series_resample_does_not_evenly_divide_day():
    index = pd.date_range('2012-01-02 00:00:00', '2012-01-02 01:00:00', freq='min')
    index = index.union(pd.date_range('2012-01-02 06:00:00', '2012-01-02 08:00:00', freq='min'))
    s = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(s, npartitions=5)
    expected = s.resample('57min').mean()
    result = ds.resample('57min').mean().compute()
    assert_eq(result, expected)