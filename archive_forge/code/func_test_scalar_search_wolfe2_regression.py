from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_scalar_search_wolfe2_regression(self):

    def phi(alpha):
        if alpha < 1:
            return -3 * np.pi / 2 * (alpha - 1)
        else:
            return np.cos(3 * np.pi / 2 * alpha - np.pi)

    def derphi(alpha):
        if alpha < 1:
            return -3 * np.pi / 2
        else:
            return -3 * np.pi / 2 * np.sin(3 * np.pi / 2 * alpha - np.pi)
    s, _, _, _ = ls.scalar_search_wolfe2(phi, derphi)
    assert s < 1.5