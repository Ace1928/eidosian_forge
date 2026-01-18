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
def test_query_pairs_eps(kdtree_type):
    spacing = np.sqrt(2)
    x_range = np.linspace(0, 3 * spacing, 4)
    y_range = np.linspace(0, 3 * spacing, 4)
    xy_array = [(xi, yi) for xi in x_range for yi in y_range]
    tree = kdtree_type(xy_array)
    pairs_eps = tree.query_pairs(r=spacing, eps=0.1)
    pairs = tree.query_pairs(r=spacing * 1.01)
    assert_equal(pairs, pairs_eps)