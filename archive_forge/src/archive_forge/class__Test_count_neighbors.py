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
@KDTreeTest
class _Test_count_neighbors(count_neighbors_consistency):

    def setup_method(self):
        n = 50
        m = 2
        np.random.seed(1234)
        self.T1 = self.kdtree_type(np.random.randn(n, m), leafsize=2)
        self.T2 = self.kdtree_type(np.random.randn(n, m), leafsize=2)