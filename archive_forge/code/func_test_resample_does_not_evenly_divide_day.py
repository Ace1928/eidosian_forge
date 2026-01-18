from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_does_not_evenly_divide_day():
    import numpy as np
    index = pd.date_range('2012-01-02', '2012-02-02', freq='h')
    index = index.union(pd.date_range('2012-03-02', '2012-04-02', freq='h'))
    df = pd.DataFrame({'p': np.random.random(len(index))}, index=index)
    ddf = dd.from_pandas(df, npartitions=5)
    expected = df.resample('2D').count()
    result = ddf.resample('2D').count().compute()
    assert_eq(result, expected)