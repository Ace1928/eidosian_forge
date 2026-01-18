import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.filterwarnings('ignore:Geometry is in a geographic CRS')
def test_sjoin_nearest_inner(self):
    countries = read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    cities = read_file(geopandas.datasets.get_path('naturalearth_cities'))
    countries = countries[['geometry', 'name']].rename(columns={'name': 'country'})
    result1 = sjoin_nearest(cities, countries, distance_col='dist')
    assert result1.shape[0] == cities.shape[0]
    result2 = sjoin_nearest(cities, countries, distance_col='dist', how='inner')
    assert_geodataframe_equal(result2, result1)
    result3 = sjoin_nearest(cities, countries, distance_col='dist', how='left')
    assert_geodataframe_equal(result3, result1, check_like=True)
    result4 = sjoin_nearest(cities, countries, distance_col='dist', max_distance=1)
    assert_geodataframe_equal(result4, result1[result1['dist'] < 1], check_like=True)
    result5 = sjoin_nearest(cities, countries, distance_col='dist', max_distance=1, how='left')
    assert result5.shape[0] == cities.shape[0]
    result5 = result5.dropna()
    result5['index_right'] = result5['index_right'].astype('int64')
    assert_geodataframe_equal(result5, result4, check_like=True)