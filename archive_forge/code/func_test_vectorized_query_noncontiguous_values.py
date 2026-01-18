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
def test_vectorized_query_noncontiguous_values(self):
    np.random.seed(1234)
    qs = np.random.randn(3, 1000).T
    ds, i_s = self.kdtree.query(qs)
    for q, d, i in zip(qs, ds, i_s):
        assert_equal(self.kdtree.query(q), (d, i))