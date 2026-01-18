from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.util import hash_pandas_object
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import tm
from dask.dataframe.utils import assert_eq
def test_object_missing_values():
    s = pd.Series(['a', 'b', 'c', None])
    h1 = hash_pandas_object(s).iloc[:3]
    h2 = hash_pandas_object(s.iloc[:3])
    tm.assert_series_equal(h1, h2)