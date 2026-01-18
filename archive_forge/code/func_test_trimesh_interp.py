from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_trimesh_interp():
    """Assert that triangles are interpolated when vertex values are provided.
    """
    verts = pd.DataFrame({'x': [0, 5, 10], 'y': [0, 10, 0]})
    tris = pd.DataFrame({'v0': [0], 'v1': [1], 'v2': [2], 'val': [1]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)
    verts = pd.DataFrame({'x': [0, 5, 10], 'y': [0, 10, 0], 'z': [1, 5, 3]})
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(verts, tris)
    sol = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 4, 4, 0, 0, 0, 0], [0, 0, 0, 3, 3, 4, 4, 0, 0, 0], [0, 0, 0, 3, 3, 3, 3, 0, 0, 0], [0, 0, 2, 3, 3, 3, 3, 3, 0, 0], [0, 0, 2, 2, 2, 3, 3, 3, 0, 0], [0, 2, 2, 2, 2, 2, 3, 3, 3, 0], [0, 1, 1, 2, 2, 2, 2, 2, 3, 0], [1, 1, 1, 1, 2, 2, 2, 2, 2, 3]], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values), sol)