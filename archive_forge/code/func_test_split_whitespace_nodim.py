from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize(['func', 'expected'], [pytest.param(lambda x: x.str.split(dim=None), [[['abc', 'def'], ['spam', 'eggs', 'swallow'], ['red_blue']], [['test0', 'test1', 'test2', 'test3'], [], ['abra', 'ka', 'da', 'bra']]], id='split_full'), pytest.param(lambda x: x.str.rsplit(dim=None), [[['abc', 'def'], ['spam', 'eggs', 'swallow'], ['red_blue']], [['test0', 'test1', 'test2', 'test3'], [], ['abra', 'ka', 'da', 'bra']]], id='rsplit_full'), pytest.param(lambda x: x.str.split(dim=None, maxsplit=1), [[['abc', 'def'], ['spam', 'eggs\tswallow'], ['red_blue']], [['test0', 'test1\ntest2\n\ntest3'], [], ['abra', 'ka\nda\tbra']]], id='split_1'), pytest.param(lambda x: x.str.rsplit(dim=None, maxsplit=1), [[['abc', 'def'], ['spam\t\teggs', 'swallow'], ['red_blue']], [['test0\ntest1\ntest2', 'test3'], [], ['abra  ka\nda', 'bra']]], id='rsplit_1')])
def test_split_whitespace_nodim(dtype, func: Callable[[xr.DataArray], xr.DataArray], expected: xr.DataArray) -> None:
    values = xr.DataArray([['abc def', 'spam\t\teggs\tswallow', 'red_blue'], ['test0\ntest1\ntest2\n\ntest3', '', 'abra  ka\nda\tbra']], dims=['X', 'Y']).astype(dtype)
    expected_dtype = [[[dtype(x) for x in y] for y in z] for z in expected]
    expected_np = np.array(expected_dtype, dtype=np.object_)
    expected_da = xr.DataArray(expected_np, dims=['X', 'Y'])
    actual = func(values)
    assert actual.dtype == expected_da.dtype
    assert_equal(actual, expected_da)