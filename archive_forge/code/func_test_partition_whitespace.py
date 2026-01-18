from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_partition_whitespace(dtype) -> None:
    values = xr.DataArray([['abc def', 'spam eggs swallow', 'red_blue'], ['test0 test1 test2 test3', '', 'abra ka da bra']], dims=['X', 'Y']).astype(dtype)
    exp_part_dim_list = [[['abc', ' ', 'def'], ['spam', ' ', 'eggs swallow'], ['red_blue', '', '']], [['test0', ' ', 'test1 test2 test3'], ['', '', ''], ['abra', ' ', 'ka da bra']]]
    exp_rpart_dim_list = [[['abc', ' ', 'def'], ['spam eggs', ' ', 'swallow'], ['', '', 'red_blue']], [['test0 test1 test2', ' ', 'test3'], ['', '', ''], ['abra ka da', ' ', 'bra']]]
    exp_part_dim = xr.DataArray(exp_part_dim_list, dims=['X', 'Y', 'ZZ']).astype(dtype)
    exp_rpart_dim = xr.DataArray(exp_rpart_dim_list, dims=['X', 'Y', 'ZZ']).astype(dtype)
    res_part_dim = values.str.partition(dim='ZZ')
    res_rpart_dim = values.str.rpartition(dim='ZZ')
    assert res_part_dim.dtype == exp_part_dim.dtype
    assert res_rpart_dim.dtype == exp_rpart_dim.dtype
    assert_equal(res_part_dim, exp_part_dim)
    assert_equal(res_rpart_dim, exp_rpart_dim)