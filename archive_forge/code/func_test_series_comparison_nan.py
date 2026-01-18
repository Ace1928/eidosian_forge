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
@pytest.mark.parametrize('comparison', ['lt', 'gt', 'le', 'ge', 'ne', 'eq'])
def test_series_comparison_nan(comparison):
    s = pd.Series([1, 2, 3, 4, 5, 6, 7])
    s_nan = pd.Series([1, -1, 8, np.nan, 5, 6, 2.4])
    ds = dd.from_pandas(s, 3)
    ds_nan = dd.from_pandas(s_nan, 3)
    fill_value = 7
    comparison_pd = getattr(s, comparison)
    comparison_dd = getattr(ds, comparison)
    assert_eq(comparison_dd(ds_nan, fill_value=fill_value), comparison_pd(s_nan, fill_value=fill_value))