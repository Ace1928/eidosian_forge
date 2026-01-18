import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('dim', range(2, 6))
@pytest.mark.parametrize('shift', [False, True])
def test_single_hemisphere_handling(self, dim, shift):
    n = 10
    points = _sample_sphere(n, dim, seed=0)
    points[:, 0] = np.abs(points[:, 0])
    center = (np.arange(dim) + 1) * shift
    sv = SphericalVoronoi(points + center, center=center)
    dots = np.einsum('ij,ij->i', sv.vertices - center, sv.points[sv._simplices[:, 0]] - center)
    circumradii = np.arccos(np.clip(dots, -1, 1))
    assert np.max(circumradii) > np.pi / 2