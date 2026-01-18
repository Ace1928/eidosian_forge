import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
def test_onetree_query(kdtree_type):
    np.random.seed(0)
    n = 50
    k = 4
    points = np.random.randn(n, k)
    T = kdtree_type(points)
    check_onetree_query(T, 0.1)
    points = np.random.randn(3 * n, k)
    points[:n] *= 0.001
    points[n:2 * n] += 2
    T = kdtree_type(points)
    check_onetree_query(T, 0.1)
    check_onetree_query(T, 0.001)
    check_onetree_query(T, 1e-05)
    check_onetree_query(T, 1e-06)