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
class _Test_sorted_query_ball_point:

    def setup_method(self):
        np.random.seed(1234)
        self.x = np.random.randn(100, 1)
        self.ckdt = self.kdtree_type(self.x)

    def test_return_sorted_True(self):
        idxs_list = self.ckdt.query_ball_point(self.x, 1.0, return_sorted=True)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))
        for xi in self.x:
            idxs = self.ckdt.query_ball_point(xi, 1.0, return_sorted=True)
            assert_array_equal(idxs, sorted(idxs))

    def test_return_sorted_None(self):
        """Previous behavior was to sort the returned indices if there were
        multiple points per query but not sort them if there was a single point
        per query."""
        idxs_list = self.ckdt.query_ball_point(self.x, 1.0)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))
        idxs_list_single = [self.ckdt.query_ball_point(xi, 1.0) for xi in self.x]
        idxs_list_False = self.ckdt.query_ball_point(self.x, 1.0, return_sorted=False)
        for idxs0, idxs1 in zip(idxs_list_False, idxs_list_single):
            assert_array_equal(idxs0, idxs1)