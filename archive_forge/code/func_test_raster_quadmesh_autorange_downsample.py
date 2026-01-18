from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_autorange_downsample(array_module):
    c = ds.Canvas(plot_width=4, plot_height=2)
    da = xr.DataArray(array_module.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]]), coords=[('b', [1, 2, 3, 4]), ('a', [1, 2, 3, 4, 5, 6, 7, 8])], name='Z')
    y_coords = np.linspace(1.5, 3.5, 2)
    x_coords = np.linspace(1.5, 7.5, 4)
    out = xr.DataArray(array_module.array([[1 + 2 + 9 + 10, 3 + 4 + 11 + 12, 5 + 6 + 13 + 14, 7 + 8 + 15 + 16], [17 + 18 + 25 + 26.0, 19 + 20 + 27 + 28, 21 + 22 + 29 + 30, 23 + 24 + 31 + 32]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)