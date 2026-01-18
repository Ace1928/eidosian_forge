from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_subpixel_line_start():
    cvs = ds.Canvas(plot_width=5, plot_height=5, x_range=(1, 3), y_range=(0, 1))
    df = pd.DataFrame(dict(x=[1, 2, 3], y0=[0.0, 0.0, 0.0], y1=[0.0, 0.08, 0.04]))
    agg = cvs.line(df, 'x', ['y0', 'y1'], agg=ds.count(), axis=1)
    xcoords = axis.compute_index(axis.compute_scale_and_translate((1.0, 3), 5), 5)
    ycoords = axis.compute_index(axis.compute_scale_and_translate((0, 1), 5), 5)
    sol = np.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[ycoords, xcoords], dims=['y', 'x'])
    assert_eq_xr(agg, out)