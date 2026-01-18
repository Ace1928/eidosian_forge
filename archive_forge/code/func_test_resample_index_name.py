from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_index_name():
    from datetime import datetime, timedelta
    import numpy as np
    date_today = datetime.now()
    days = pd.date_range(date_today, date_today + timedelta(20), freq='D')
    data = np.random.randint(1, high=100, size=len(days))
    df = pd.DataFrame({'date': days, 'values': data})
    df = df.set_index('date')
    ddf = dd.from_pandas(df, npartitions=4)
    assert ddf.resample('D').mean().head().index.name == 'date'