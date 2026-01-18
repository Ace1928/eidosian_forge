from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('high', [9, 10, 99, 100])
@pytest.mark.parametrize('low', [0])
def test_uniform_diagonal_points(low, high):
    bounds = (low, high)
    x_range, y_range = (bounds, bounds)
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    n = width * height
    df = pd.DataFrame({'time': np.ones(n, dtype='i4'), 'x': np.array([np.arange(*x_range, dtype='f8')] * width).flatten(), 'y': np.array([np.arange(*y_range, dtype='f8')] * height).flatten()})
    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    diagonal = agg.data.diagonal(0)
    assert sum(diagonal) == n
    assert abs(bounds[1] - bounds[0]) % 2 == abs(diagonal[1] / high - diagonal[0] / high)
    assert_eq_ndarray(agg.x_range, (low, high), close=True)
    assert_eq_ndarray(agg.y_range, (low, high), close=True)