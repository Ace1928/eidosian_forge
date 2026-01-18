from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
def test_where_last(df):
    out = xr.DataArray([[16, 6], [11, 1]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'), 'reverse')), out)
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.last('f32'))), out)