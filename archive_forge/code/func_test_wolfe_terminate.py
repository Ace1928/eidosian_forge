from numpy.testing import (assert_equal, assert_array_almost_equal,
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
import numpy as np
def test_wolfe_terminate(self):

    def phi(s):
        count[0] += 1
        return -s + 0.05 * s ** 2

    def derphi(s):
        count[0] += 1
        return -1 + 0.05 * 2 * s
    for func in [ls.scalar_search_wolfe1, ls.scalar_search_wolfe2]:
        count = [0]
        r = func(phi, derphi, phi(0), None, derphi(0))
        assert r[0] is not None, (r, func)
        assert count[0] <= 2 + 2, (count, func)
        assert_wolfe(r[0], phi, derphi, err_msg=str(func))