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
def test_where_min(df):
    out = xr.DataArray([[20, 10], [15, 5]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'), 'reverse')), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'), 'reverse')), out)
    out = xr.DataArray([[0, 10], [5, 15]], coords=coords, dims=dims)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i32'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('i64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f64'))), out)
    assert_eq_xr(c.points(df, 'x', 'y', ds.where(ds.min('f32'))), out)