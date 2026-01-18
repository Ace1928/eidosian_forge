from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
@pytest.mark.parametrize('method', ['split', 'rsplit'])
def test_str_accessor_split_expand(method):

    def call(obj, *args, **kwargs):
        return getattr(obj.str, method)(*args, **kwargs)
    s = pd.Series(['a b c d', 'aa bb cc dd', 'aaa bbb ccc dddd'], index=['row1', 'row2', 'row3'])
    ds = dd.from_pandas(s, npartitions=2)
    for n in [1, 2, 3]:
        assert_eq(call(s, n=n, expand=True), call(ds, n=n, expand=True))
    with pytest.raises(NotImplementedError) as info:
        call(ds, expand=True)
    assert 'n=' in str(info.value)
    s = pd.Series(['a,bcd,zz,f', 'aabb,ccdd,z,kk', 'aaabbb,cccdddd,l,pp'])
    ds = dd.from_pandas(s, npartitions=2)
    for n in [1, 2, 3]:
        assert_eq(call(s, pat=',', n=n, expand=True), call(ds, pat=',', n=n, expand=True))