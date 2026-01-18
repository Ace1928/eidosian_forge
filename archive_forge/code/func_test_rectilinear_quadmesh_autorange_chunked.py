from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
def test_rectilinear_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=8, plot_height=6)
    da = xr.DataArray(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), coords=[('b', [1, 2, 3]), ('a', [1, 2, 3, 8])], name='Z').chunk({'a': 2, 'b': 3})
    y_coords = np.linspace(0.75, 3.25, 6)
    x_coords = np.linspace(1.125, 9.875, 8)
    out = xr.DataArray(np.array([[3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0], [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0], [11.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0], [11.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0], [19.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0], [19.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 3.5), close=True)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (0.5, 10.5), close=True)
    assert_eq_ndarray(res.y_range, (0.5, 3.5), close=True)