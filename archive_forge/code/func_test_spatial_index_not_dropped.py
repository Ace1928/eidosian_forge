import pytest
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
import dask.dataframe as dd
@pytest.mark.skipif(not spatialpandas, reason='spatialpandas not installed')
def test_spatial_index_not_dropped():
    df = GeoDataFrame({'some_geom': MultiPolygonArray([[[[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]]], [[[0, 2, 1, 2, 1, 3, 0, 3, 0, 2]]]]), 'other': [23, 45]})
    assert df.some_geom.array._sindex is None
    sindex = df.some_geom.array.sindex
    assert sindex is not None
    glyph = ds.glyphs.polygon.PolygonGeom('some_geom')
    agg = ds.count()
    df2, _ = ds.core._bypixel_sanitise(df, glyph, agg)
    assert df2.columns == ['some_geom']
    assert df2.some_geom.array._sindex == df.some_geom.array._sindex