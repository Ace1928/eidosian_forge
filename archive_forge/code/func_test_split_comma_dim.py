from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize(['func', 'expected'], [pytest.param(lambda x: x.str.split(sep=',', dim='ZZ'), [[['abc', 'def', '', ''], ['spam', '', 'eggs', 'swallow'], ['red_blue', '', '', '']], [['test0', 'test1', 'test2', 'test3'], ['', '', '', ''], ['abra', 'ka', 'da', 'bra']]], id='split_full'), pytest.param(lambda x: x.str.rsplit(sep=',', dim='ZZ'), [[['', '', 'abc', 'def'], ['spam', '', 'eggs', 'swallow'], ['', '', '', 'red_blue']], [['test0', 'test1', 'test2', 'test3'], ['', '', '', ''], ['abra', 'ka', 'da', 'bra']]], id='rsplit_full'), pytest.param(lambda x: x.str.split(sep=',', dim='ZZ', maxsplit=1), [[['abc', 'def'], ['spam', ',eggs,swallow'], ['red_blue', '']], [['test0', 'test1,test2,test3'], ['', ''], ['abra', 'ka,da,bra']]], id='split_1'), pytest.param(lambda x: x.str.rsplit(sep=',', dim='ZZ', maxsplit=1), [[['abc', 'def'], ['spam,,eggs', 'swallow'], ['', 'red_blue']], [['test0,test1,test2', 'test3'], ['', ''], ['abra,ka,da', 'bra']]], id='rsplit_1'), pytest.param(lambda x: x.str.split(sep=',', dim='ZZ', maxsplit=10), [[['abc', 'def', '', ''], ['spam', '', 'eggs', 'swallow'], ['red_blue', '', '', '']], [['test0', 'test1', 'test2', 'test3'], ['', '', '', ''], ['abra', 'ka', 'da', 'bra']]], id='split_10'), pytest.param(lambda x: x.str.rsplit(sep=',', dim='ZZ', maxsplit=10), [[['', '', 'abc', 'def'], ['spam', '', 'eggs', 'swallow'], ['', '', '', 'red_blue']], [['test0', 'test1', 'test2', 'test3'], ['', '', '', ''], ['abra', 'ka', 'da', 'bra']]], id='rsplit_10')])
def test_split_comma_dim(dtype, func: Callable[[xr.DataArray], xr.DataArray], expected: xr.DataArray) -> None:
    values = xr.DataArray([['abc,def', 'spam,,eggs,swallow', 'red_blue'], ['test0,test1,test2,test3', '', 'abra,ka,da,bra']], dims=['X', 'Y']).astype(dtype)
    expected_dtype = [[[dtype(x) for x in y] for y in z] for z in expected]
    expected_np = np.array(expected_dtype, dtype=np.object_)
    expected_da = xr.DataArray(expected_np, dims=['X', 'Y', 'ZZ']).astype(dtype)
    actual = func(values)
    assert actual.dtype == expected_da.dtype
    assert_equal(actual, expected_da)