from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.skipif(not sp, reason='spatialpandas not installed')
def test_points_geometry_point():
    axis = ds.core.LinearAxis()
    lincoords = axis.compute_index(axis.compute_scale_and_translate((0.0, 2.0), 3), 3)
    df = sp.GeoDataFrame({'geom': pd.array([[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]], dtype='Point[float64]'), 'v': [1, 2, 2, 3, 3, 3]})
    cvs = ds.Canvas(plot_width=3, plot_height=3)
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    sol = np.array([[1, nan, nan], [2, 2, nan], [3, 3, 3]], dtype='float64')
    out = xr.DataArray(sol, coords=[lincoords, lincoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)
    assert df.geom.array._sindex is None
    df.geom.array.sindex
    agg = cvs.points(df, geometry='geom', agg=ds.sum('v'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 2), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)