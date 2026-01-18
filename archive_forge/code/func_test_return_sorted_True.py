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
def test_return_sorted_True(self):
    idxs_list = self.ckdt.query_ball_point(self.x, 1.0, return_sorted=True)
    for idxs in idxs_list:
        assert_array_equal(idxs, sorted(idxs))
    for xi in self.x:
        idxs = self.ckdt.query_ball_point(xi, 1.0, return_sorted=True)
        assert_array_equal(idxs, sorted(idxs))