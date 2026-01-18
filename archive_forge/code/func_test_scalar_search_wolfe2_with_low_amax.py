from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_scalar_search_wolfe2_with_low_amax(self):

    def phi(alpha):
        return (alpha - 5) ** 2

    def derphi(alpha):
        return 2 * (alpha - 5)
    s, _, _, _ = assert_warns(LineSearchWarning, ls.scalar_search_wolfe2, phi, derphi, amax=0.001)
    assert s is None