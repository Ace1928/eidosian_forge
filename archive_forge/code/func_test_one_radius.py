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
def test_one_radius(self):
    r = 0.2
    assert_equal(self.T1.count_neighbors(self.T2, r), np.sum([len(l) for l in self.T1.query_ball_tree(self.T2, r)]))