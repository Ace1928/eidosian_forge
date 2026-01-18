from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_agg():
    index = pd.date_range('2000-01-01', '2000-02-15', freq='h')
    ps = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(ps, npartitions=2)
    assert_eq(ds.resample('10min').agg('mean'), ps.resample('10min').agg('mean'))
    assert_eq(ds.resample('10min').agg(['mean', 'min']), ps.resample('10min').agg(['mean', 'min']))