from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skip_no_sindex
class TestPygeosInterface:

    def setup_method(self):
        data = {'geom': [Point(x, y) for x, y in zip(range(5), range(5))] + [box(10, 10, 20, 20)]}
        self.df = GeoDataFrame(data, geometry='geom')
        self.expected_size = len(data['geom'])

    @pytest.mark.parametrize('test_geom, expected', (((-1, -1, -0.5, -0.5), []), ((-0.5, -0.5, 0.5, 0.5), [0]), ((0, 0, 1, 1), [0, 1]), ((0, 0), [0])))
    def test_intersection_bounds_tuple(self, test_geom, expected):
        """Tests the `intersection` method with valid inputs."""
        res = list(self.df.sindex.intersection(test_geom))
        assert_array_equal(res, expected)

    @pytest.mark.parametrize('test_geom', ((-1, -1, -0.5), -0.5, None, Point(0, 0)))
    def test_intersection_invalid_bounds_tuple(self, test_geom):
        """Tests the `intersection` method with invalid inputs."""
        if compat.USE_PYGEOS:
            with pytest.raises(TypeError):
                self.df.sindex.intersection(test_geom)
        else:
            with pytest.raises((TypeError, Exception)):
                self.df.sindex.intersection(test_geom)

    @pytest.mark.parametrize('predicate, test_geom, expected', ((None, box(-1, -1, -0.5, -0.5), []), (None, box(-0.5, -0.5, 0.5, 0.5), [0]), (None, box(0, 0, 1, 1), [0, 1]), (None, LineString([(0, 1), (1, 0)]), [0, 1]), ('intersects', box(-1, -1, -0.5, -0.5), []), ('intersects', box(-0.5, -0.5, 0.5, 0.5), [0]), ('intersects', box(0, 0, 1, 1), [0, 1]), ('intersects', LineString([(0, 1), (1, 0)]), []), ('within', box(0.25, 0.28, 0.75, 0.75), []), ('within', box(0, 0, 10, 10), []), ('within', box(11, 11, 12, 12), [5]), ('within', LineString([(0, 1), (1, 0)]), []), ('contains', box(0, 0, 1, 1), []), ('contains', box(0, 0, 1.001, 1.001), [1]), ('contains', box(0.5, 0.5, 1.5, 1.5), [1]), ('contains', box(-1, -1, 2, 2), [0, 1]), ('contains', LineString([(0, 1), (1, 0)]), []), ('touches', box(-1, -1, 0, 0), [0]), ('touches', box(-0.5, -0.5, 1.5, 1.5), []), ('contains', box(10, 10, 20, 20), [5]), ('covers', box(-0.5, -0.5, 1, 1), [0, 1]), ('covers', box(0.001, 0.001, 0.99, 0.99), []), ('covers', box(0, 0, 1, 1), [0, 1]), ('contains_properly', box(0, 0, 1, 1), []), ('contains_properly', box(0, 0, 1.001, 1.001), [1]), ('contains_properly', box(0.5, 0.5, 1.001, 1.001), [1]), ('contains_properly', box(0.5, 0.5, 1.5, 1.5), [1]), ('contains_properly', box(-1, -1, 2, 2), [0, 1]), ('contains_properly', box(10, 10, 20, 20), [])))
    def test_query(self, predicate, test_geom, expected):
        """Tests the `query` method with valid inputs and valid predicates."""
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    def test_query_invalid_geometry(self):
        """Tests the `query` method with invalid geometry."""
        with pytest.raises(TypeError):
            self.df.sindex.query('notavalidgeom')

    @pytest.mark.parametrize('test_geom, expected_value', [(None, []), (GeometryCollection(), []), (Point(), []), (MultiPolygon(), []), (Polygon(), [])])
    def test_query_empty_geometry(self, test_geom, expected_value):
        """Tests the `query` method with empty geometry."""
        res = self.df.sindex.query(test_geom)
        assert_array_equal(res, expected_value)

    def test_query_invalid_predicate(self):
        """Tests the `query` method with invalid predicates."""
        test_geom = box(-1, -1, -0.5, -0.5)
        with pytest.raises(ValueError):
            self.df.sindex.query(test_geom, predicate='test')

    @pytest.mark.parametrize('sort, expected', ((True, [[0, 0, 0], [0, 1, 2]]), (False, [[0, 0, 0], [0, 1, 2]])))
    def test_query_sorting(self, sort, expected):
        """Check that results from `query` don't depend on the
        order of geometries.
        """
        test_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
        tree_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        expected = [0, 1, 2]
        tree_df = geopandas.GeoDataFrame(geometry=tree_polys)
        test_df = geopandas.GeoDataFrame(geometry=test_polys)
        test_geo = test_df.geometry.values[0]
        res = tree_df.sindex.query(test_geo, sort=sort)
        assert sorted(res) == sorted(expected)
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if sort is False:
                pytest.xfail('rtree results are known to be unordered, see https://github.com/geopandas/geopandas/issues/1337\nExpected:\n {}\n'.format(expected) + 'Got:\n {}\n'.format(res.tolist()))
            raise e

    @pytest.mark.parametrize('predicate, test_geom, expected', ((None, [(-1, -1, -0.5, -0.5)], [[], []]), (None, [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]), (None, [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('intersects', [(-1, -1, -0.5, -0.5)], [[], []]), ('intersects', [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]), ('intersects', [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('intersects', [(-1, -1, -0.5, -0.5), (-0.5, -0.5, 0.5, 0.5)], [[1], [0]]), ('intersects', [(-1, -1, 1, 1), (-0.5, -0.5, 0.5, 0.5)], [[0, 0, 1], [0, 1, 0]]), ('within', [(0.25, 0.28, 0.75, 0.75)], [[], []]), ('within', [(0, 0, 10, 10)], [[], []]), ('within', [(11, 11, 12, 12)], [[0], [5]]), ('contains', [(0, 0, 1, 1)], [[], []]), ('contains', [(0, 0, 1.001, 1.001)], [[0], [1]]), ('contains', [(0.5, 0.5, 1.001, 1.001)], [[0], [1]]), ('contains', [(0.5, 0.5, 1.5, 1.5)], [[0], [1]]), ('contains', [(-1, -1, 2, 2)], [[0, 0], [0, 1]]), ('contains', [(10, 10, 20, 20)], [[0], [5]]), ('touches', [(-1, -1, 0, 0)], [[0], [0]]), ('touches', [(-0.5, -0.5, 1.5, 1.5)], [[], []]), ('covers', [(-0.5, -0.5, 1, 1)], [[0, 0], [0, 1]]), ('covers', [(0.001, 0.001, 0.99, 0.99)], [[], []]), ('covers', [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('contains_properly', [(0, 0, 1, 1)], [[], []]), ('contains_properly', [(0, 0, 1.001, 1.001)], [[0], [1]]), ('contains_properly', [(0.5, 0.5, 1.001, 1.001)], [[0], [1]]), ('contains_properly', [(0.5, 0.5, 1.5, 1.5)], [[0], [1]]), ('contains_properly', [(-1, -1, 2, 2)], [[0, 0], [0, 1]]), ('contains_properly', [(10, 10, 20, 20)], [[], []])))
    def test_query_bulk(self, predicate, test_geom, expected):
        """Tests the `query_bulk` method with valid
        inputs and valid predicates.
        """
        test_geom = geopandas.GeoSeries([box(*geom) for geom in test_geom], index=range(len(test_geom)))
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize('test_geoms, expected_value', [([GeometryCollection()], [[], []]), ([GeometryCollection(), None], [[], []]), ([None], [[], []]), ([None, box(-0.5, -0.5, 0.5, 0.5), None], [[1], [0]])])
    def test_query_bulk_empty_geometry(self, test_geoms, expected_value):
        """Tests the `query_bulk` method with an empty geometry."""
        test_geoms = geopandas.GeoSeries(test_geoms, index=range(len(test_geoms)))
        res = self.df.sindex.query(test_geoms)
        assert_array_equal(res, expected_value)

    def test_query_bulk_empty_input_array(self):
        """Tests the `query_bulk` method with an empty input array."""
        test_array = np.array([], dtype=object)
        expected_value = [[], []]
        res = self.df.sindex.query(test_array)
        assert_array_equal(res, expected_value)

    def test_query_bulk_invalid_input_geometry(self):
        """
        Tests the `query_bulk` method with invalid input for the `geometry` parameter.
        """
        test_array = 'notanarray'
        with pytest.raises(TypeError):
            self.df.sindex.query(test_array)

    def test_query_bulk_invalid_predicate(self):
        """Tests the `query_bulk` method with invalid predicates."""
        test_geom_bounds = (-1, -1, -0.5, -0.5)
        test_predicate = 'test'
        test_geom = geopandas.GeoSeries([box(*test_geom_bounds)], index=['0'])
        with pytest.raises(ValueError):
            self.df.sindex.query(test_geom.geometry, predicate=test_predicate)

    @pytest.mark.parametrize('predicate, test_geom, expected', ((None, (-1, -1, -0.5, -0.5), [[], []]), ('intersects', (-1, -1, -0.5, -0.5), [[], []]), ('contains', (-1, -1, 1, 1), [[0], [0]])))
    def test_query_bulk_input_type(self, predicate, test_geom, expected):
        """Tests that query_bulk can accept a GeoSeries, GeometryArray or
        numpy array.
        """
        test_geom = geopandas.GeoSeries([box(*test_geom)], index=['0'])
        res = self.df.sindex.query(test_geom, predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query(test_geom.geometry, predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query(test_geom.geometry.values, predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query(test_geom.geometry.values.to_numpy(), predicate=predicate)
        assert_array_equal(res, expected)
        res = self.df.sindex.query(test_geom.geometry.values.to_numpy(), predicate=predicate)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize('sort, expected', ((True, [[0, 0, 0], [0, 1, 2]]), (False, [[0, 0, 0], [0, 1, 2]])))
    def test_query_bulk_sorting(self, sort, expected):
        """Check that results from `query_bulk` don't depend
        on the order of geometries.
        """
        test_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])])
        tree_polys = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        tree_df = geopandas.GeoDataFrame(geometry=tree_polys)
        test_df = geopandas.GeoDataFrame(geometry=test_polys)
        res = tree_df.sindex.query(test_df.geometry, sort=sort)
        assert sorted(res[0]) == sorted(expected[0])
        assert sorted(res[1]) == sorted(expected[1])
        try:
            assert_array_equal(res, expected)
        except AssertionError as e:
            if sort is False:
                pytest.xfail('rtree results are known to be unordered, see https://github.com/geopandas/geopandas/issues/1337\nExpected:\n {}\n'.format(expected) + 'Got:\n {}\n'.format(res.tolist()))
            raise e

    @pytest.mark.skipif(compat.USE_PYGEOS or compat.USE_SHAPELY_20, reason='RTree supports sindex.nearest with different behaviour')
    def test_rtree_nearest_warns(self):
        df = geopandas.GeoDataFrame({'geometry': []})
        with pytest.warns(FutureWarning, match='sindex.nearest using the rtree backend'):
            df.sindex.nearest((0, 0, 1, 1), num_results=2)

    @pytest.mark.skipif(compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and (not compat.PYGEOS_GE_010)), reason='PyGEOS < 0.10 does not support sindex.nearest')
    def test_pygeos_error(self):
        df = geopandas.GeoDataFrame({'geometry': []})
        with pytest.raises(NotImplementedError, match='requires pygeos >= 0.10'):
            df.sindex.nearest(None)

    @pytest.mark.skipif(not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)), reason='PyGEOS >= 0.10 is required to test sindex.nearest')
    @pytest.mark.parametrize('return_all', [True, False])
    @pytest.mark.parametrize('geometry,expected', [([0.25, 0.25], [[0], [0]]), ([0.75, 0.75], [[0], [1]])])
    def test_nearest_single(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({'geometry': geoms})
        p = Point(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)
        p = mod.points(geometry)
        res = df.sindex.nearest(p, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)), reason='PyGEOS >= 0.10 is required to test sindex.nearest')
    @pytest.mark.parametrize('return_all', [True, False])
    @pytest.mark.parametrize('geometry,expected', [([(1, 1), (0, 0)], [[0, 1], [1, 0]]), ([(1, 1), (0.25, 1)], [[0, 1], [1, 1]])])
    def test_nearest_multi(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({'geometry': geoms})
        ps = [Point(p) for p in geometry]
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)
        ps = mod.points(geometry)
        res = df.sindex.nearest(ps, return_all=return_all)
        assert_array_equal(res, expected)
        s = geopandas.GeoSeries(ps)
        res = df.sindex.nearest(s, return_all=return_all)
        assert_array_equal(res, expected)
        x, y = zip(*geometry)
        ga = geopandas.points_from_xy(x, y)
        res = df.sindex.nearest(ga, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)), reason='PyGEOS >= 0.10 is required to test sindex.nearest')
    @pytest.mark.parametrize('return_all', [True, False])
    @pytest.mark.parametrize('geometry,expected', [(None, [[], []]), ([None], [[], []])])
    def test_nearest_none(self, geometry, expected, return_all):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({'geometry': geoms})
        res = df.sindex.nearest(geometry, return_all=return_all)
        assert_array_equal(res, expected)

    @pytest.mark.skipif(not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)), reason='PyGEOS >= 0.10 is required to test sindex.nearest')
    @pytest.mark.parametrize('return_distance', [True, False])
    @pytest.mark.parametrize('return_all,max_distance,expected', [(True, None, ([[0, 0, 1], [0, 1, 5]], [sqrt(0.5), sqrt(0.5), sqrt(50)])), (False, None, ([[0, 1], [0, 5]], [sqrt(0.5), sqrt(50)])), (True, 1, ([[0, 0], [0, 1]], [sqrt(0.5), sqrt(0.5)])), (False, 1, ([[0], [0]], [sqrt(0.5)]))])
    def test_nearest_max_distance(self, expected, max_distance, return_all, return_distance):
        geoms = mod.points(np.arange(10), np.arange(10))
        df = geopandas.GeoDataFrame({'geometry': geoms})
        ps = [Point(0.5, 0.5), Point(0, 10)]
        res = df.sindex.nearest(ps, return_all=return_all, max_distance=max_distance, return_distance=return_distance)
        if return_distance:
            assert_array_equal(res[0], expected[0])
            assert_array_equal(res[1], expected[1])
        else:
            assert_array_equal(res, expected[0])

    @pytest.mark.skipif(not compat.USE_SHAPELY_20, reason='shapely >= 2.0 is required to test sindex.nearest with parameter exclusive')
    @pytest.mark.parametrize('return_distance', [True, False])
    @pytest.mark.parametrize('return_all,max_distance,exclusive,expected', [(False, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])), (False, None, True, ([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3]], 5 * [sqrt(2)])), (True, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])), (True, None, True, ([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], 8 * [sqrt(2)])), (False, 1.1, True, ([[1, 2, 5], [5, 5, 1]], 3 * [1])), (True, 1.1, True, ([[1, 2, 5, 5], [5, 5, 1, 2]], 4 * [1]))])
    def test_nearest_exclusive(self, expected, max_distance, return_all, return_distance, exclusive):
        geoms = mod.points(np.arange(5), np.arange(5))
        if max_distance:
            geoms = np.append(geoms, [Point(1, 2)])
        df = geopandas.GeoDataFrame({'geometry': geoms})
        ps = geoms
        res = df.sindex.nearest(ps, return_all=return_all, max_distance=max_distance, return_distance=return_distance, exclusive=exclusive)
        if return_distance:
            assert_array_equal(res[0], expected[0])
            assert_array_equal(res[1], expected[1])
        else:
            assert_array_equal(res, expected[0])

    @pytest.mark.skipif(compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and (not compat.PYGEOS_GE_010)), reason='sindex.nearest exclusive parameter requires shapely >= 2.0')
    def test_nearest_exclusive_unavailable(self):
        from shapely.geometry import Point
        geoms = [Point((x, y)) for x, y in zip(np.arange(5), np.arange(5))]
        df = geopandas.GeoDataFrame(geometry=geoms)
        with pytest.raises(NotImplementedError, match='requires shapely >= 2.0'):
            df.sindex.nearest(geoms, exclusive=True)

    def test_empty_tree_geometries(self):
        """Tests building sindex with interleaved empty geometries."""
        geoms = [Point(0, 0), None, Point(), Point(1, 1), Point()]
        df = geopandas.GeoDataFrame(geometry=geoms)
        assert df.sindex.query(Point(1, 1))[0] == 3

    def test_size(self):
        """Tests the `size` property."""
        assert self.df.sindex.size == self.expected_size

    def test_len(self):
        """Tests the `__len__` method of spatial indexes."""
        assert len(self.df.sindex) == self.expected_size

    def test_is_empty(self):
        """Tests the `is_empty` property."""
        empty = geopandas.GeoSeries([], dtype=object)
        assert empty.sindex.is_empty
        empty = geopandas.GeoSeries([None])
        assert empty.sindex.is_empty
        empty = geopandas.GeoSeries([Point()])
        assert empty.sindex.is_empty
        non_empty = geopandas.GeoSeries([Point(0, 0)])
        assert not non_empty.sindex.is_empty

    @pytest.mark.parametrize('predicate, expected_shape', [(None, (2, 471)), ('intersects', (2, 213)), ('within', (2, 213)), ('contains', (2, 0)), ('overlaps', (2, 0)), ('crosses', (2, 0)), ('touches', (2, 0))])
    def test_integration_natural_earth(self, predicate, expected_shape):
        """Tests output sizes for the naturalearth datasets."""
        world = read_file(datasets.get_path('naturalearth_lowres'))
        capitals = read_file(datasets.get_path('naturalearth_cities'))
        res = world.sindex.query(capitals.geometry, predicate)
        assert res.shape == expected_shape