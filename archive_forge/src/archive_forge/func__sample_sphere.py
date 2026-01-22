import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def _sample_sphere(n, dim, seed=None):
    rng = np.random.RandomState(seed=seed)
    points = rng.randn(n, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points