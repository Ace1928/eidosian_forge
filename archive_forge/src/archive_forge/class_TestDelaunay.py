import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
class TestDelaunay:
    """
    Check that triangulation works.

    """

    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.Delaunay, masked_array)

    def test_array_with_nans_fails(self):
        points_with_nan = np.array([(0, 0), (0, 1), (1, 1), (1, np.nan)], dtype=np.float64)
        assert_raises(ValueError, qhull.Delaunay, points_with_nan)

    def test_nd_simplex(self):
        for nd in range(2, 8):
            points = np.zeros((nd + 1, nd))
            for j in range(nd):
                points[j, j] = 1.0
            points[-1, :] = 1.0
            tri = qhull.Delaunay(points)
            tri.simplices.sort()
            assert_equal(tri.simplices, np.arange(nd + 1, dtype=int)[None, :])
            assert_equal(tri.neighbors, -1 + np.zeros(nd + 1, dtype=int)[None, :])

    def test_2d_square(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        assert_equal(tri.simplices, [[1, 3, 2], [3, 1, 0]])
        assert_equal(tri.neighbors, [[-1, -1, 1], [-1, -1, 0]])

    def test_duplicate_points(self):
        x = np.array([0, 1, 0, 1], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        xp = np.r_[x, x]
        yp = np.r_[y, y]
        qhull.Delaunay(np.c_[x, y])
        qhull.Delaunay(np.c_[xp, yp])

    def test_pathological(self):
        points = DATASETS['pathological-1']
        tri = qhull.Delaunay(points)
        assert_equal(tri.points[tri.simplices].max(), points.max())
        assert_equal(tri.points[tri.simplices].min(), points.min())
        points = DATASETS['pathological-2']
        tri = qhull.Delaunay(points)
        assert_equal(tri.points[tri.simplices].max(), points.max())
        assert_equal(tri.points[tri.simplices].min(), points.min())

    def test_joggle(self):
        points = np.random.rand(10, 2)
        points = np.r_[points, points]
        tri = qhull.Delaunay(points, qhull_options='QJ Qbb Pp')
        assert_array_equal(np.unique(tri.simplices.ravel()), np.arange(len(points)))

    def test_coplanar(self):
        points = np.random.rand(10, 2)
        points = np.r_[points, points]
        tri = qhull.Delaunay(points)
        assert_(len(np.unique(tri.simplices.ravel())) == len(points) // 2)
        assert_(len(tri.coplanar) == len(points) // 2)
        assert_(len(np.unique(tri.coplanar[:, 2])) == len(points) // 2)
        assert_(np.all(tri.vertex_to_simplex >= 0))

    def test_furthest_site(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
        tri = qhull.Delaunay(points, furthest_site=True)
        expected = np.array([(1, 4, 0), (4, 2, 0)])
        assert_array_equal(tri.simplices, expected)

    @pytest.mark.parametrize('name', sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        chunks, opts = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)
        obj = qhull.Delaunay(chunks[0], incremental=True, qhull_options=opts)
        for chunk in chunks[1:]:
            obj.add_points(chunk)
        obj2 = qhull.Delaunay(points)
        obj3 = qhull.Delaunay(chunks[0], incremental=True, qhull_options=opts)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0), restart=True)
        if name.startswith('pathological'):
            assert_array_equal(np.unique(obj.simplices.ravel()), np.arange(points.shape[0]))
            assert_array_equal(np.unique(obj2.simplices.ravel()), np.arange(points.shape[0]))
        else:
            assert_unordered_tuple_list_equal(obj.simplices, obj2.simplices, tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(obj2.simplices, obj3.simplices, tpl=sorted_tuple)