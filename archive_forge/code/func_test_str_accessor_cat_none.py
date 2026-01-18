from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
def test_str_accessor_cat_none():
    s = pd.Series(['a', 'a', 'b', 'b', 'c', np.nan], name='foo')
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(ds.str.cat(), s.str.cat())
    assert_eq(ds.str.cat(na_rep='-'), s.str.cat(na_rep='-'))
    assert_eq(ds.str.cat(sep='_', na_rep='-'), s.str.cat(sep='_', na_rep='-'))