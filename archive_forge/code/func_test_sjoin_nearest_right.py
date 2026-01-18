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
@pytest.mark.parametrize('geo_left, geo_right, expected_left, expected_right, distances', [([Point(0, 0), Point(1, 1)], [Point(1, 1)], [1], [0], [0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0)], [1, 0], [0, 1], [0, 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0), Point(0, 0)], [1, 0, 0], [0, 1, 2], [0, 0, 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0), Point(2, 2)], [1, 0, 1], [0, 1, 2], [0, 0, math.sqrt(2)]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0.25, 1)], [1, 1], [0, 1], [0, 0.75]), ([Point(0, 0), Point(1, 1)], [Point(-10, -10), Point(100, 100)], [0, 1], [0, 1], [math.sqrt(10 ** 2 + 10 ** 2), math.sqrt(99 ** 2 + 99 ** 2)]), ([Point(0, 0), Point(1, 1)], [Point(x, y) for x, y in zip(np.arange(10), np.arange(10))], [0, 1] + [1] * 8, list(range(10)), [0, 0] + [np.sqrt(x ** 2 + x ** 2) for x in np.arange(1, 9)]), ([Point(0, 0), Point(1, 1), Point(0, 0)], [Point(1.1, 1.1), Point(0, 0)], [1, 0, 2], [0, 1, 1], [np.sqrt(0.1 ** 2 + 0.1 ** 2), 0, 0])])
def test_sjoin_nearest_right(self, geo_left, geo_right, expected_left: Sequence[int], expected_right: Sequence[int], distances: Sequence[float]):
    left = geopandas.GeoDataFrame({'geometry': geo_left})
    right = geopandas.GeoDataFrame({'geometry': geo_right})
    expected_gdf = right.iloc[expected_right].copy()
    expected_gdf['index_left'] = expected_left
    expected_gdf = expected_gdf[['index_left', 'geometry']]
    joined = sjoin_nearest(left, right, how='right')
    assert_geodataframe_equal(expected_gdf, joined)
    expected_gdf['distance_col'] = np.array(distances, dtype=float)
    joined = sjoin_nearest(left, right, how='right', distance_col='distance_col')
    assert_geodataframe_equal(expected_gdf, joined)