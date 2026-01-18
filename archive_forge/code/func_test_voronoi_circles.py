import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_voronoi_circles(self):
    sv = SphericalVoronoi(self.points)
    for vertex in sv.vertices:
        distances = distance.cdist(sv.points, np.array([vertex]))
        closest = np.array(sorted(distances)[0:3])
        assert_almost_equal(closest[0], closest[1], 7, str(vertex))
        assert_almost_equal(closest[0], closest[2], 7, str(vertex))