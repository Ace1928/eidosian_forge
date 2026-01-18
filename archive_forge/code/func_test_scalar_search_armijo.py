from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_scalar_search_armijo(self):
    for name, phi, derphi, old_phi0 in self.scalar_iter():
        s, phi1 = ls.scalar_search_armijo(phi, phi(0), derphi(0))
        assert_fp_equal(phi1, phi(s), name)
        assert_armijo(s, phi, err_msg=f'{name} {old_phi0:g}')