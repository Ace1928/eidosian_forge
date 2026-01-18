from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testCopySize(self):
    n = [0, 0, 1, 0, 0]
    m = make_mask(n)
    m2 = make_mask(m)
    assert_(m is m2)
    m3 = make_mask(m, copy=True)
    assert_(m is not m3)
    x1 = np.arange(5)
    y1 = array(x1, mask=m)
    assert_(y1._data is not x1)
    assert_(allequal(x1, y1._data))
    assert_(y1._mask is m)
    y1a = array(y1, copy=0)
    assert_(y1a._mask.__array_interface__ == y1._mask.__array_interface__)
    y2 = array(x1, mask=m3, copy=0)
    assert_(y2._mask is m3)
    assert_(y2[2] is masked)
    y2[2] = 9
    assert_(y2[2] is not masked)
    assert_(y2._mask is m3)
    assert_(allequal(y2.mask, 0))
    y2a = array(x1, mask=m, copy=1)
    assert_(y2a._mask is not m)
    assert_(y2a[2] is masked)
    y2a[2] = 9
    assert_(y2a[2] is not masked)
    assert_(y2a._mask is not m)
    assert_(allequal(y2a.mask, 0))
    y3 = array(x1 * 1.0, mask=m)
    assert_(filled(y3).dtype is (x1 * 1.0).dtype)
    x4 = arange(4)
    x4[2] = masked
    y4 = resize(x4, (8,))
    assert_(eq(concatenate([x4, x4]), y4))
    assert_(eq(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0]))
    y5 = repeat(x4, (2, 2, 2, 2), axis=0)
    assert_(eq(y5, [0, 0, 1, 1, 2, 2, 3, 3]))
    y6 = repeat(x4, 2, axis=0)
    assert_(eq(y5, y6))