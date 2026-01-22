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
class _Test_random_ball_largep_issue9890(ball_consistency):
    tol = 1e-13

    def setup_method(self):
        n = 1000
        m = 2
        np.random.seed(123)
        self.data = np.random.randint(100, 1000, size=(n, m))
        self.T = self.kdtree_type(self.data)
        self.x = self.data
        self.p = 100
        self.eps = 0
        self.d = 10