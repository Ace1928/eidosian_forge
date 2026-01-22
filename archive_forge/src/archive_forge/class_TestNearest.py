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
@pytest.mark.skipif(not TEST_NEAREST, reason='PyGEOS >= 0.10.0 must be installed and activated via the geopandas.compat module to test sjoin_nearest')
class TestNearest:

    @pytest.mark.parametrize('how_kwargs', ({}, {'how': 'inner'}, {'how': 'left'}, {'how': 'right'}))
    def test_allowed_hows(self, how_kwargs):
        left = geopandas.GeoDataFrame({'geometry': []})
        right = geopandas.GeoDataFrame({'geometry': []})
        sjoin_nearest(left, right, **how_kwargs)

    @pytest.mark.parametrize('how', ('outer', 'abcde'))
    def test_invalid_hows(self, how: str):
        left = geopandas.GeoDataFrame({'geometry': []})
        right = geopandas.GeoDataFrame({'geometry': []})
        with pytest.raises(ValueError, match='`how` was'):
            sjoin_nearest(left, right, how=how)

    @pytest.mark.parametrize('distance_col', (None, 'distance'))
    def test_empty_right_df_how_left(self, distance_col: str):
        left = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({'geometry': []})
        joined = sjoin_nearest(left, right, how='left', distance_col=distance_col)
        assert_geoseries_equal(joined['geometry'], left['geometry'])
        assert joined['index_right'].isna().all()
        if distance_col is not None:
            assert joined[distance_col].isna().all()

    @pytest.mark.parametrize('distance_col', (None, 'distance'))
    def test_empty_right_df_how_right(self, distance_col: str):
        left = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({'geometry': []})
        joined = sjoin_nearest(left, right, how='right', distance_col=distance_col)
        assert joined.empty
        if distance_col is not None:
            assert distance_col in joined

    @pytest.mark.parametrize('how', ['inner', 'left'])
    @pytest.mark.parametrize('distance_col', (None, 'distance'))
    def test_empty_left_df(self, how, distance_col: str):
        right = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        left = geopandas.GeoDataFrame({'geometry': []})
        joined = sjoin_nearest(left, right, how=how, distance_col=distance_col)
        assert joined.empty
        if distance_col is not None:
            assert distance_col in joined

    @pytest.mark.parametrize('distance_col', (None, 'distance'))
    def test_empty_left_df_how_right(self, distance_col: str):
        right = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        left = geopandas.GeoDataFrame({'geometry': []})
        joined = sjoin_nearest(left, right, how='right', distance_col=distance_col)
        assert_geoseries_equal(joined['geometry'], right['geometry'])
        assert joined['index_left'].isna().all()
        if distance_col is not None:
            assert joined[distance_col].isna().all()

    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_empty_join_due_to_max_distance(self, how):
        left = geopandas.GeoDataFrame({'geometry': [Point(0, 0)]})
        right = geopandas.GeoDataFrame({'geometry': [Point(1, 1), Point(2, 2)]})
        joined = sjoin_nearest(left, right, how=how, max_distance=1, distance_col='distances')
        expected = left.copy()
        expected['index_right'] = [np.nan]
        expected['distances'] = [np.nan]
        if how == 'inner':
            expected = expected.dropna()
            expected['index_right'] = expected['index_right'].astype('int64')
        assert_geodataframe_equal(joined, expected)

    def test_empty_join_due_to_max_distance_how_right(self):
        left = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({'geometry': [Point(2, 2)]})
        joined = sjoin_nearest(left, right, how='right', max_distance=1, distance_col='distances')
        expected = right.copy()
        expected['index_left'] = [np.nan]
        expected['distances'] = [np.nan]
        expected = expected[['index_left', 'geometry', 'distances']]
        assert_geodataframe_equal(joined, expected)

    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_max_distance(self, how):
        left = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({'geometry': [Point(1, 1), Point(2, 2)]})
        joined = sjoin_nearest(left, right, how=how, max_distance=1, distance_col='distances')
        expected = left.copy()
        expected['index_right'] = [np.nan, 0]
        expected['distances'] = [np.nan, 0]
        if how == 'inner':
            expected = expected.dropna()
            expected['index_right'] = expected['index_right'].astype('int64')
        assert_geodataframe_equal(joined, expected)

    def test_max_distance_how_right(self):
        left = geopandas.GeoDataFrame({'geometry': [Point(1, 1), Point(2, 2)]})
        right = geopandas.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        joined = sjoin_nearest(left, right, how='right', max_distance=1, distance_col='distances')
        expected = right.copy()
        expected['index_left'] = [np.nan, 0]
        expected['distances'] = [np.nan, 0]
        expected = expected[['index_left', 'geometry', 'distances']]
        assert_geodataframe_equal(joined, expected)

    @pytest.mark.parametrize('how', ['inner', 'left'])
    @pytest.mark.parametrize('geo_left, geo_right, expected_left, expected_right, distances', [([Point(0, 0), Point(1, 1)], [Point(1, 1)], [0, 1], [0, 0], [math.sqrt(2), 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0)], [0, 1], [1, 0], [0, 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0), Point(0, 0)], [0, 0, 1], [1, 2, 0], [0, 0, 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0, 0), Point(2, 2)], [0, 1], [1, 0], [0, 0]), ([Point(0, 0), Point(1, 1)], [Point(1, 1), Point(0.25, 1)], [0, 1], [1, 0], [math.sqrt(0.25 ** 2 + 1), 0]), ([Point(0, 0), Point(1, 1)], [Point(-10, -10), Point(100, 100)], [0, 1], [0, 0], [math.sqrt(10 ** 2 + 10 ** 2), math.sqrt(11 ** 2 + 11 ** 2)]), ([Point(0, 0), Point(1, 1)], [Point(x, y) for x, y in zip(np.arange(10), np.arange(10))], [0, 1], [0, 1], [0, 0]), ([Point(0, 0), Point(1, 1), Point(0, 0)], [Point(1.1, 1.1), Point(0, 0)], [0, 1, 2], [1, 0, 1], [0, np.sqrt(0.1 ** 2 + 0.1 ** 2), 0])])
    def test_sjoin_nearest_left(self, geo_left, geo_right, expected_left: Sequence[int], expected_right: Sequence[int], distances: Sequence[float], how):
        left = geopandas.GeoDataFrame({'geometry': geo_left})
        right = geopandas.GeoDataFrame({'geometry': geo_right})
        expected_gdf = left.iloc[expected_left].copy()
        expected_gdf['index_right'] = expected_right
        joined = sjoin_nearest(left, right, how=how)
        check_like = how == 'inner'
        assert_geodataframe_equal(expected_gdf, joined, check_like=check_like)
        expected_gdf['distance_col'] = np.array(distances, dtype=float)
        joined = sjoin_nearest(left, right, how=how, distance_col='distance_col')
        assert_geodataframe_equal(expected_gdf, joined, check_like=check_like)

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
    expected_index_uncapped = [1, 3, 3, 1, 2] if compat.PANDAS_GE_22 else [1, 1, 3, 3, 2]

    @pytest.mark.skipif(not compat.USE_SHAPELY_20, reason='shapely >= 2.0 is required to run sjoin_nearestwith parameter `exclusive` set')
    @pytest.mark.parametrize('max_distance,expected', [(None, expected_index_uncapped), (1.1, [3, 3, 1, 2])])
    def test_sjoin_nearest_exclusive(self, max_distance, expected):
        geoms = shapely.points(np.arange(3), np.arange(3))
        geoms = np.append(geoms, [Point(1, 2)])
        df = geopandas.GeoDataFrame({'geometry': geoms})
        result = df.sjoin_nearest(df, max_distance=max_distance, distance_col='dist', exclusive=True)
        assert_series_equal(result['index_right'].reset_index(drop=True), pd.Series(expected), check_names=False)
        if max_distance:
            assert result['dist'].max() <= max_distance