import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.mark.filterwarnings('ignore:GeoSeries crs mismatch:UserWarning')
def test_overlay_nybb(how):
    polydf = read_file(geopandas.datasets.get_path('nybb'))
    polydf2 = read_file(os.path.join(DATA, 'nybb_qgis', 'polydf2.shp'))
    result = overlay(polydf, polydf2, how=how)
    cols = ['BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area', 'value1', 'value2']
    if how == 'difference':
        cols = cols[:-2]
    if how == 'identity':
        expected = read_file(os.path.join(DATA, 'nybb_qgis', 'qgis-union.shp'))
    else:
        expected = read_file(os.path.join(DATA, 'nybb_qgis', 'qgis-{0}.shp'.format(how)))
    if how == 'union':
        expected = expected.drop([24, 27])
        expected.reset_index(inplace=True, drop=True)
    expected = expected[expected.is_valid]
    expected.reset_index(inplace=True, drop=True)
    if how == 'identity':
        expected = expected[expected.BoroCode.notnull()].copy()
    expected = expected.sort_values(cols).reset_index(drop=True)
    result = result.sort_values(cols).reset_index(drop=True)
    if how in ('union', 'identity'):
        assert result.columns[-1] == 'geometry'
        assert len(result.columns) == len(expected.columns)
        result = result.reindex(columns=expected.columns)
    kwargs = {}
    pd.testing.assert_series_equal(result.geometry.area, expected.geometry.area, **kwargs)
    pd.testing.assert_frame_equal(result.geometry.bounds, expected.geometry.bounds, **kwargs)
    if how == 'symmetric_difference':
        expected.loc[9, 'geometry'] = None
        result.loc[9, 'geometry'] = None
    if how == 'union':
        expected.loc[24, 'geometry'] = None
        result.loc[24, 'geometry'] = None
    assert_geodataframe_equal(result, expected, normalize=True, check_crs=False, check_column_type=False, check_less_precise=True)