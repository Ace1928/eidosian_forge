from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_trimesh_winding_detect():
    """Assert that CCW windings get converted to CW.
    """
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4], 'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3, 4]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([[0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)
    verts = pd.DataFrame({'x': [4, 1, 5, 5, 5, 4], 'y': [4, 5, 5, 5, 4, 4]})
    tris = pd.DataFrame({'v0': [0, 3], 'v1': [2, 5], 'v2': [1, 4], 'val': [3.0, 4.0]})
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.trimesh(verts, tris)
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)