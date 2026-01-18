import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_num_vertices(self):
    sv = SphericalVoronoi(self.points)
    expected = self.points.shape[0] * 2 - 4
    actual = sv.vertices.shape[0]
    assert_equal(actual, expected)