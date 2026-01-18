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
@pytest.mark.parametrize('func', ['skew', 'kurtosis'])
def test_skew_kurt_numeric_only_false(func):
    pytest.importorskip('scipy.stats')
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2010, i, 1) for i in range(1, 8)]})
    ddf = dd.from_pandas(df, npartitions=2)
    ctx = pytest.raises(TypeError, match='does not support|does not implement')
    with ctx:
        getattr(df, func)(numeric_only=False)
    with ctx:
        getattr(ddf, func)(numeric_only=False)
    if PANDAS_GE_150 and (not PANDAS_GE_200):
        ctx = pytest.warns(FutureWarning, match='default value')
    elif not PANDAS_GE_150:
        ctx = pytest.warns(FutureWarning, match='nuisance columns')
    with ctx:
        getattr(df, func)()
    with ctx:
        getattr(ddf, func)()