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
def test_pivot_table_errors():
    df = pd.DataFrame({'A': np.random.choice(list('abc'), size=10), 'B': np.random.randn(10), 'C': pd.Categorical(np.random.choice(list('abc'), size=10))})
    ddf = dd.from_pandas(df, 2)
    msg = "'index' must be the name of an existing column"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index=['A'], columns='C', values='B')
    assert msg in str(err.value)
    msg = "'columns' must be the name of an existing column"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns=['C'], values='B')
    assert msg in str(err.value)
    msg = "'values' must refer to an existing column or columns"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns='C', values=[['B']])
    assert msg in str(err.value)
    msg = "aggfunc must be either 'mean', 'sum', 'count', 'first', 'last'"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns='C', values='B', aggfunc=['sum'])
    assert msg in str(err.value)
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns='C', values='B', aggfunc='xx')
    assert msg in str(err.value)
    ddf['C'] = ddf.C.cat.as_unknown()
    msg = "'columns' must have known categories"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns='C', values=['B'])
    assert msg in str(err.value)
    df = pd.DataFrame({'A': np.random.choice(list('abc'), size=10), 'B': np.random.randn(10), 'C': np.random.choice(list('abc'), size=10)})
    ddf = dd.from_pandas(df, 2)
    msg = "'columns' must be category dtype"
    with pytest.raises(ValueError) as err:
        dd.pivot_table(ddf, index='A', columns='C', values='B')
    assert msg in str(err.value)