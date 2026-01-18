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
def test_vectorized_query(self):
    d, i = self.kdtree.query(np.zeros((2, 4, 3)))
    assert_equal(np.shape(d), (2, 4))
    assert_equal(np.shape(i), (2, 4))