from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testArithmetic(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    a2d = array([[1, 2], [0, 4]])
    a2dm = masked_array(a2d, [[0, 0], [1, 0]])
    assert_(eq(a2d * a2d, a2d * a2dm))
    assert_(eq(a2d + a2d, a2d + a2dm))
    assert_(eq(a2d - a2d, a2d - a2dm))
    for s in [(12,), (4, 3), (2, 6)]:
        x = x.reshape(s)
        y = y.reshape(s)
        xm = xm.reshape(s)
        ym = ym.reshape(s)
        xf = xf.reshape(s)
        assert_(eq(-x, -xm))
        assert_(eq(x + y, xm + ym))
        assert_(eq(x - y, xm - ym))
        assert_(eq(x * y, xm * ym))
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_(eq(x / y, xm / ym))
        assert_(eq(a10 + y, a10 + ym))
        assert_(eq(a10 - y, a10 - ym))
        assert_(eq(a10 * y, a10 * ym))
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_(eq(a10 / y, a10 / ym))
        assert_(eq(x + a10, xm + a10))
        assert_(eq(x - a10, xm - a10))
        assert_(eq(x * a10, xm * a10))
        assert_(eq(x / a10, xm / a10))
        assert_(eq(x ** 2, xm ** 2))
        assert_(eq(abs(x) ** 2.5, abs(xm) ** 2.5))
        assert_(eq(x ** y, xm ** ym))
        assert_(eq(np.add(x, y), add(xm, ym)))
        assert_(eq(np.subtract(x, y), subtract(xm, ym)))
        assert_(eq(np.multiply(x, y), multiply(xm, ym)))
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_(eq(np.divide(x, y), divide(xm, ym)))