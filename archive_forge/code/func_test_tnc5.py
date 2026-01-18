import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_tnc5(self):
    fg, x, bounds = (self.fg5, [0, 0], [(-1.5, 4), (-3, 3)])
    xopt = [-0.5471975511965976, -1.5471975511965976]
    x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, messages=optimize._tnc.MSG_NONE, maxfun=200)
    assert_allclose(self.f5(x), self.f5(xopt), atol=1e-08, err_msg='TNC failed with status: ' + optimize._tnc.RCSTRINGS[rc])