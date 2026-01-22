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
class _Test_random(ConsistencyTests):

    def setup_method(self):
        self.n = 100
        self.m = 4
        np.random.seed(1234)
        self.data = np.random.randn(self.n, self.m)
        self.kdtree = self.kdtree_type(self.data, leafsize=2)
        self.x = np.random.randn(self.m)
        self.d = 0.2
        self.k = 10