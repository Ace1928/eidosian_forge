from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_str_accessor(df_ddf):
    df, ddf = df_ddf
    assert 'upper' in dir(ddf.str_col.str)
    assert 'upper' in dir(ddf.string_col.str)
    assert 'upper' in dir(ddf.index.str)
    assert 'get_dummies' not in dir(ddf.str_col.str)
    assert not hasattr(ddf.str_col.str, 'get_dummies')
    assert_eq(ddf.str_col.str.upper(), df.str_col.str.upper())
    assert set(ddf.str_col.str.upper().dask) == set(ddf.str_col.str.upper().dask)
    assert_eq(ddf.string_col.str.upper(), df.string_col.str.upper())
    assert set(ddf.string_col.str.upper().dask) == set(ddf.string_col.str.upper().dask)
    assert_eq(ddf.index.str.upper(), df.index.str.upper())
    assert set(ddf.index.str.upper().dask) == set(ddf.index.str.upper().dask)
    ctx = contextlib.nullcontext()
    if pyarrow_strings_enabled():
        df.str_col = to_pyarrow_string(df.str_col)
        if not PANDAS_GE_210:
            ctx = pytest.warns(pd.errors.PerformanceWarning, match='Falling back on a non-pyarrow')
    assert_eq(ddf.str_col.str.contains('a'), df.str_col.str.contains('a'))
    assert_eq(ddf.string_col.str.contains('a'), df.string_col.str.contains('a'))
    assert set(ddf.str_col.str.contains('a').dask) == set(ddf.str_col.str.contains('a').dask)
    with ctx:
        expected = df.str_col.str.contains('d', case=False)
    assert_eq(ddf.str_col.str.contains('d', case=False), expected)
    assert set(ddf.str_col.str.contains('d', case=False).dask) == set(ddf.str_col.str.contains('d', case=False).dask)
    for na in [True, False]:
        assert_eq(ddf.str_col.str.contains('a', na=na), df.str_col.str.contains('a', na=na))
        assert set(ddf.str_col.str.contains('a', na=na).dask) == set(ddf.str_col.str.contains('a', na=na).dask)
    for regex in [True, False]:
        assert_eq(ddf.str_col.str.contains('a', regex=regex), df.str_col.str.contains('a', regex=regex))
        assert set(ddf.str_col.str.contains('a', regex=regex).dask) == set(ddf.str_col.str.contains('a', regex=regex).dask)