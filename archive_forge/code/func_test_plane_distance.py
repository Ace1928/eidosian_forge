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