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
class TestUtilities:
    """
    Check that utility functions work.

    """

    def test_find_simplex(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        assert_equal(tri.simplices, [[1, 3, 2], [3, 1, 0]])
        for p in [(0.25, 0.25, 1), (0.75, 0.75, 0), (0.3, 0.2, 1)]:
            i = tri.find_simplex(p[:2])
            assert_equal(i, p[2], err_msg=f'{p!r}')
            j = qhull.tsearch(tri, p[:2])
            assert_equal(i, j)

    def test_plane_distance(self):
        x = np.array([(0, 0), (1, 1), (1, 0), (0.99189033, 0.37674127), (0.99440079, 0.45182168)], dtype=np.float64)
        p = np.array([0.99966555, 0.15685619], dtype=np.float64)
        tri = qhull.Delaunay(x)
        z = tri.lift_points(x)
        pz = tri.lift_points(p)
        dist = tri.plane_distance(p)
        for j, v in enumerate(tri.simplices):
            x1 = z[v[0]]
            x2 = z[v[1]]
            x3 = z[v[2]]
            n = np.cross(x1 - x3, x2 - x3)
            n /= np.sqrt(np.dot(n, n))
            n *= -np.sign(n[2])
            d = np.dot(n, pz - x3)
            assert_almost_equal(dist[j], d)

    def test_convex_hull(self):
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        tri = qhull.Delaunay(points)
        assert_equal(tri.convex_hull, [[3, 2], [1, 2], [1, 0], [3, 0]])

    def test_volume_area(self):
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        hull = qhull.ConvexHull(points)
        assert_allclose(hull.volume, 1.0, rtol=1e-14, err_msg='Volume of cube is incorrect')
        assert_allclose(hull.area, 6.0, rtol=1e-14, err_msg='Area of cube is incorrect')

    def test_random_volume_area(self):
        points = np.array([(0.362568364506, 0.472712355305, 0.347003084477), (0.733731893414, 0.634480295684, 0.950513180209), (0.511239955611, 0.876839441267, 0.418047827863), (0.0765906233393, 0.527373281342, 0.6509863541), (0.146694972056, 0.596725793348, 0.894860986685), (0.513808585741, 0.069576205858, 0.530890338876), (0.512343805118, 0.663537132612, 0.037689295973), (0.47282965018, 0.462176697655, 0.14061843691), (0.240584597123, 0.778660020591, 0.722913476339), (0.951271745935, 0.967000673944, 0.890661319684)])
        hull = qhull.ConvexHull(points)
        assert_allclose(hull.volume, 0.14562013, rtol=1e-07, err_msg='Volume of random polyhedron is incorrect')
        assert_allclose(hull.area, 1.6670425, rtol=1e-07, err_msg='Area of random polyhedron is incorrect')

    def test_incremental_volume_area_random_input(self):
        """Test that incremental mode gives the same volume/area as
        non-incremental mode and incremental mode with restart"""
        nr_points = 20
        dim = 3
        points = np.random.random((nr_points, dim))
        inc_hull = qhull.ConvexHull(points[:dim + 1, :], incremental=True)
        inc_restart_hull = qhull.ConvexHull(points[:dim + 1, :], incremental=True)
        for i in range(dim + 1, nr_points):
            hull = qhull.ConvexHull(points[:i + 1, :])
            inc_hull.add_points(points[i:i + 1, :])
            inc_restart_hull.add_points(points[i:i + 1, :], restart=True)
            assert_allclose(hull.volume, inc_hull.volume, rtol=1e-07)
            assert_allclose(hull.volume, inc_restart_hull.volume, rtol=1e-07)
            assert_allclose(hull.area, inc_hull.area, rtol=1e-07)
            assert_allclose(hull.area, inc_restart_hull.area, rtol=1e-07)

    def _check_barycentric_transforms(self, tri, err_msg='', unit_cube=False, unit_cube_tol=0):
        """Check that a triangulation has reasonable barycentric transforms"""
        vertices = tri.points[tri.simplices]
        sc = 1 / (tri.ndim + 1.0)
        centroids = vertices.sum(axis=1) * sc

        def barycentric_transform(tr, x):
            r = tr[:, -1, :]
            Tinv = tr[:, :-1, :]
            return np.einsum('ijk,ik->ij', Tinv, x - r)
        eps = np.finfo(float).eps
        c = barycentric_transform(tri.transform, centroids)
        with np.errstate(invalid='ignore'):
            ok = np.isnan(c).all(axis=1) | (abs(c - sc) / sc < 0.1).all(axis=1)
        assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
        q = vertices[:, :-1, :] - vertices[:, -1, None, :]
        volume = np.array([np.linalg.det(q[k, :, :]) for k in range(tri.nsimplex)])
        ok = np.isfinite(tri.transform[:, 0, 0]) | (volume < np.sqrt(eps))
        assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
        j = tri.find_simplex(centroids)
        ok = (j != -1) | np.isnan(tri.transform[:, 0, 0])
        assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')
        if unit_cube:
            at_boundary = (centroids <= unit_cube_tol).any(axis=1)
            at_boundary |= (centroids >= 1 - unit_cube_tol).any(axis=1)
            ok = (j != -1) | at_boundary
            assert_(ok.all(), f'{err_msg} {np.nonzero(~ok)}')

    def test_degenerate_barycentric_transforms(self):
        data = np.load(os.path.join(os.path.dirname(__file__), 'data', 'degenerate_pointset.npz'))
        points = data['c']
        data.close()
        tri = qhull.Delaunay(points)
        bad_count = np.isnan(tri.transform[:, 0, 0]).sum()
        assert_(bad_count < 23, bad_count)
        self._check_barycentric_transforms(tri)

    @pytest.mark.slow
    def test_more_barycentric_transforms(self):
        eps = np.finfo(float).eps
        npoints = {2: 70, 3: 11, 4: 5, 5: 3}
        for ndim in range(2, 6):
            x = np.linspace(0, 1, npoints[ndim])
            grid = np.c_[list(map(np.ravel, np.broadcast_arrays(*np.ix_(*[x] * ndim))))].T
            err_msg = 'ndim=%d' % ndim
            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True)
            np.random.seed(1234)
            m = np.random.rand(grid.shape[0]) < 0.2
            grid[m, :] += 2 * eps * (np.random.rand(*grid[m, :].shape) - 0.5)
            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True, unit_cube_tol=2 * eps)
            tri = qhull.Delaunay(np.r_[grid, grid])
            self._check_barycentric_transforms(tri, err_msg=err_msg, unit_cube=True, unit_cube_tol=2 * eps)