import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_tnc1c(self):
    x, bounds = ([-2, 1], ([-np.inf, None], [-1.5, None]))
    xopt = [1, 1]
    x, nf, rc = optimize.fmin_tnc(self.f1, x, fprime=self.g1, bounds=bounds, messages=optimize._tnc.MSG_NONE, maxfun=200)
    assert_allclose(self.f1(x), self.f1(xopt), atol=1e-08, err_msg='TNC failed with status: ' + optimize._tnc.RCSTRINGS[rc])