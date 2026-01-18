from __future__ import annotations
import numpy as np
from xarray import DataArray
from datashader.datashape import dshape
from datashader.utils import Dispatcher, apply, calc_res, isreal, orient_array
def test_orient_array():
    x = [5, 7]
    y = [0, 1]
    z = np.array([[0, 1], [2, 3]])
    dims = ('y', 'x')
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z)
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1])
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[:, ::-1])
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    arr = orient_array(xarr)
    assert np.array_equal(arr, z[::-1, ::-1])