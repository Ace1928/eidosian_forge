import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_tnc3(self):
    fg, x, bounds = (self.fg3, [10, 1], ([-np.inf, None], [0.0, None]))
    xopt = [0, 0]
    x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, messages=optimize._tnc.MSG_NONE, maxfun=200)
    assert_allclose(self.f3(x), self.f3(xopt), atol=1e-08, err_msg='TNC failed with status: ' + optimize._tnc.RCSTRINGS[rc])