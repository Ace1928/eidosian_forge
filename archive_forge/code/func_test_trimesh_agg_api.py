from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_trimesh_agg_api():
    """Assert that the trimesh aggregation API properly handles weights on the simplices."""
    pts = pd.DataFrame({'x': [1, 3, 4, 3, 3], 'y': [2, 1, 2, 1, 4]}, columns=['x', 'y'])
    tris = pd.DataFrame({'n1': [4, 1], 'n2': [1, 4], 'n3': [2, 0], 'weight': [0.83231525, 1.3053126]}, columns=['n1', 'n2', 'n3', 'weight'])
    cvs = ds.Canvas(x_range=(0, 10), y_range=(0, 10))
    agg = cvs.trimesh(pts, tris, agg=ds.mean('weight'))
    assert agg.shape == (600, 600)
    assert_eq_ndarray(agg.x_range, (0, 10), close=True)
    assert_eq_ndarray(agg.y_range, (0, 10), close=True)