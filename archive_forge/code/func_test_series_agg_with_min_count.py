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
@pytest.mark.parametrize('method', ['sum', 'prod', 'product'])
@pytest.mark.parametrize('min_count', [0, 9])
def test_series_agg_with_min_count(method, min_count):
    df = pd.DataFrame([[1]], columns=['a'])
    ddf = dd.from_pandas(df, npartitions=1)
    func = getattr(ddf['a'], method)
    result = func(min_count=min_count).compute()
    if min_count == 0:
        assert result == 1
    else:
        assert result is np.nan or pd.isna(result)