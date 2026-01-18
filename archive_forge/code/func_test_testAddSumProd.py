from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testAddSumProd(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    assert_(eq(np.add.reduce(x), add.reduce(x)))
    assert_(eq(np.add.accumulate(x), add.accumulate(x)))
    assert_(eq(4, sum(array(4), axis=0)))
    assert_(eq(4, sum(array(4), axis=0)))
    assert_(eq(np.sum(x, axis=0), sum(x, axis=0)))
    assert_(eq(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0)))
    assert_(eq(np.sum(x, 0), sum(x, 0)))
    assert_(eq(np.prod(x, axis=0), product(x, axis=0)))
    assert_(eq(np.prod(x, 0), product(x, 0)))
    assert_(eq(np.prod(filled(xm, 1), axis=0), product(xm, axis=0)))
    if len(s) > 1:
        assert_(eq(np.concatenate((x, y), 1), concatenate((xm, ym), 1)))
        assert_(eq(np.add.reduce(x, 1), add.reduce(x, 1)))
        assert_(eq(np.sum(x, 1), sum(x, 1)))
        assert_(eq(np.prod(x, 1), product(x, 1)))