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
class _Test_two_random_trees(two_trees_consistency):

    def setup_method(self):
        n = 50
        m = 4
        np.random.seed(1234)
        self.data1 = np.random.randn(n, m)
        self.T1 = self.kdtree_type(self.data1, leafsize=2)
        self.data2 = np.random.randn(n, m)
        self.T2 = self.kdtree_type(self.data2, leafsize=2)
        self.p = 2.0
        self.eps = 0
        self.d = 0.2