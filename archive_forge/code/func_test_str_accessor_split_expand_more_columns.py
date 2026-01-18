from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.mark.xfail(reason='Need to pad columns')
def test_str_accessor_split_expand_more_columns():
    s = pd.Series(['a b c d', 'aa', 'aaa bbb ccc dddd'])
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(s.str.split(n=3, expand=True), ds.str.split(n=3, expand=True))
    s = pd.Series(['a b c', 'aa bb cc', 'aaa bbb ccc'])
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(ds.str.split(n=10, expand=True), s.str.split(n=10, expand=True))