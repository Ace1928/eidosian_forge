from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_scalar_search_wolfe1(self):
    c = 0
    for name, phi, derphi, old_phi0 in self.scalar_iter():
        c += 1
        s, phi1, phi0 = ls.scalar_search_wolfe1(phi, derphi, phi(0), old_phi0, derphi(0))
        assert_fp_equal(phi0, phi(0), name)
        assert_fp_equal(phi1, phi(s), name)
        assert_wolfe(s, phi, derphi, err_msg=name)
    assert c > 3