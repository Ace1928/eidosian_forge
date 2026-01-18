import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_tnc38(self):
    fg, x, bounds = (self.fg38, np.array([-3, -1, -3, -1]), [(-10, 10)] * 4)
    xopt = [1] * 4
    x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, messages=optimize._tnc.MSG_NONE, maxfun=200)
    assert_allclose(self.f38(x), self.f38(xopt), atol=1e-08, err_msg='TNC failed with status: ' + optimize._tnc.RCSTRINGS[rc])