from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
def test_pivot_table_dtype():
    df = pd.DataFrame({'A': list('AABB'), 'B': pd.Categorical(list('ABAB')), 'C': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, 2)
    res = dd.pivot_table(ddf, index='A', columns='B', values='C', aggfunc='count')
    exp_index = pd.CategoricalIndex(['A', 'B'], name='B')
    exp = pd.Series([np.float64] * 2, index=exp_index)
    tm.assert_series_equal(res.dtypes, exp)
    exp = pd.pivot_table(df, index='A', columns='B', values='C', aggfunc='count', observed=False).astype(np.float64)
    assert_eq(res, exp)