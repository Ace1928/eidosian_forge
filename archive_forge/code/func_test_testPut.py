from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testPut(self):
    d = arange(5)
    n = [0, 0, 0, 1, 1]
    m = make_mask(n)
    m2 = m.copy()
    x = array(d, mask=m)
    assert_(x[3] is masked)
    assert_(x[4] is masked)
    x[[1, 4]] = [10, 40]
    assert_(x._mask is m)
    assert_(x[3] is masked)
    assert_(x[4] is not masked)
    assert_(eq(x, [0, 10, 2, -1, 40]))
    x = array(d, mask=m2, copy=True)
    x.put([0, 1, 2], [-1, 100, 200])
    assert_(x._mask is not m2)
    assert_(x[3] is masked)
    assert_(x[4] is masked)
    assert_(eq(x, [-1, 100, 200, 0, 0]))