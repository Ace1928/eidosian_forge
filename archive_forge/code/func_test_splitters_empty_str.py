from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_splitters_empty_str(dtype) -> None:
    values = xr.DataArray([['', '', ''], ['', '', '']], dims=['X', 'Y']).astype(dtype)
    targ_partition_dim = xr.DataArray([[['', '', ''], ['', '', ''], ['', '', '']], [['', '', ''], ['', '', ''], ['', '', '']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    targ_partition_none_list = [[['', '', ''], ['', '', ''], ['', '', '']], [['', '', ''], ['', '', ''], ['', '', '', '']]]
    targ_partition_none_list = [[[dtype(x) for x in y] for y in z] for z in targ_partition_none_list]
    targ_partition_none_np = np.array(targ_partition_none_list, dtype=np.object_)
    del targ_partition_none_np[-1, -1][-1]
    targ_partition_none = xr.DataArray(targ_partition_none_np, dims=['X', 'Y'])
    targ_split_dim = xr.DataArray([[[''], [''], ['']], [[''], [''], ['']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    targ_split_none = xr.DataArray(np.array([[[], [], []], [[], [], ['']]], dtype=np.object_), dims=['X', 'Y'])
    del targ_split_none.data[-1, -1][-1]
    res_partition_dim = values.str.partition(dim='ZZ')
    res_rpartition_dim = values.str.rpartition(dim='ZZ')
    res_partition_none = values.str.partition(dim=None)
    res_rpartition_none = values.str.rpartition(dim=None)
    res_split_dim = values.str.split(dim='ZZ')
    res_rsplit_dim = values.str.rsplit(dim='ZZ')
    res_split_none = values.str.split(dim=None)
    res_rsplit_none = values.str.rsplit(dim=None)
    res_dummies = values.str.rsplit(dim='ZZ')
    assert res_partition_dim.dtype == targ_partition_dim.dtype
    assert res_rpartition_dim.dtype == targ_partition_dim.dtype
    assert res_partition_none.dtype == targ_partition_none.dtype
    assert res_rpartition_none.dtype == targ_partition_none.dtype
    assert res_split_dim.dtype == targ_split_dim.dtype
    assert res_rsplit_dim.dtype == targ_split_dim.dtype
    assert res_split_none.dtype == targ_split_none.dtype
    assert res_rsplit_none.dtype == targ_split_none.dtype
    assert res_dummies.dtype == targ_split_dim.dtype
    assert_equal(res_partition_dim, targ_partition_dim)
    assert_equal(res_rpartition_dim, targ_partition_dim)
    assert_equal(res_partition_none, targ_partition_none)
    assert_equal(res_rpartition_none, targ_partition_none)
    assert_equal(res_split_dim, targ_split_dim)
    assert_equal(res_rsplit_dim, targ_split_dim)
    assert_equal(res_split_none, targ_split_none)
    assert_equal(res_rsplit_none, targ_split_none)
    assert_equal(res_dummies, targ_split_dim)