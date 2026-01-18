from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_unknown_divisions_error():
    df = pd.DataFrame({'x': [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=2, sort=False).clear_divisions()
    try:
        ddf.x.resample('1m').mean()
        assert False
    except ValueError as e:
        assert 'divisions' in str(e)