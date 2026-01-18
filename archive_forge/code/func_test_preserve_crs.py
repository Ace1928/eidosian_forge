import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_preserve_crs(dfs, how):
    df1, df2 = dfs
    result = overlay(df1, df2, how=how)
    assert result.crs is None
    crs = 'epsg:4326'
    df1.crs = crs
    df2.crs = crs
    result = overlay(df1, df2, how=how)
    assert result.crs == crs