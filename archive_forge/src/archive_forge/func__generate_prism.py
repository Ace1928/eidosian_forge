import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
def _generate_prism(n, axis):
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    bottom = np.vstack([-np.ones(n), np.cos(thetas), np.sin(thetas)]).T
    top = np.vstack([+np.ones(n), np.cos(thetas), np.sin(thetas)]).T
    P = np.concatenate((bottom, top))
    return np.roll(P, axis, axis=1)