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
def test_single_query_multiple_neighbors(self):
    s = 23
    kk = self.kdtree.n + s
    d, i = self.kdtree.query([0, 0, 0], k=kk)
    assert_equal(np.shape(d), (kk,))
    assert_equal(np.shape(i), (kk,))
    assert_(np.all(~np.isfinite(d[-s:])))
    assert_(np.all(i[-s:] == self.kdtree.n))