from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testAverage2(self):
    w1 = [0, 1, 1, 1, 1, 0]
    w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
    x = arange(6)
    assert_(allclose(average(x, axis=0), 2.5))
    assert_(allclose(average(x, axis=0, weights=w1), 2.5))
    y = array([arange(6), 2.0 * arange(6)])
    assert_(allclose(average(y, None), np.add.reduce(np.arange(6)) * 3.0 / 12.0))
    assert_(allclose(average(y, axis=0), np.arange(6) * 3.0 / 2.0))
    assert_(allclose(average(y, axis=1), [average(x, axis=0), average(x, axis=0) * 2.0]))
    assert_(allclose(average(y, None, weights=w2), 20.0 / 6.0))
    assert_(allclose(average(y, axis=0, weights=w2), [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]))
    assert_(allclose(average(y, axis=1), [average(x, axis=0), average(x, axis=0) * 2.0]))
    m1 = zeros(6)
    m2 = [0, 0, 1, 1, 0, 0]
    m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
    m4 = ones(6)
    m5 = [0, 1, 1, 1, 1, 1]
    assert_(allclose(average(masked_array(x, m1), axis=0), 2.5))
    assert_(allclose(average(masked_array(x, m2), axis=0), 2.5))
    assert_(average(masked_array(x, m4), axis=0) is masked)
    assert_equal(average(masked_array(x, m5), axis=0), 0.0)
    assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
    z = masked_array(y, m3)
    assert_(allclose(average(z, None), 20.0 / 6.0))
    assert_(allclose(average(z, axis=0), [0.0, 1.0, 99.0, 99.0, 4.0, 7.5]))
    assert_(allclose(average(z, axis=1), [2.5, 5.0]))
    assert_(allclose(average(z, axis=0, weights=w2), [0.0, 1.0, 99.0, 99.0, 4.0, 10.0]))
    a = arange(6)
    b = arange(6) * 3
    r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
    assert_equal(shape(r1), shape(w1))
    assert_equal(r1.shape, w1.shape)
    r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
    assert_equal(shape(w2), shape(r2))
    r2, w2 = average(ones((2, 2, 3)), returned=True)
    assert_equal(shape(w2), shape(r2))
    r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
    assert_(shape(w2) == shape(r2))
    a2d = array([[1, 2], [0, 4]], float)
    a2dm = masked_array(a2d, [[0, 0], [1, 0]])
    a2da = average(a2d, axis=0)
    assert_(eq(a2da, [0.5, 3.0]))
    a2dma = average(a2dm, axis=0)
    assert_(eq(a2dma, [1.0, 3.0]))
    a2dma = average(a2dm, axis=None)
    assert_(eq(a2dma, 7.0 / 3.0))
    a2dma = average(a2dm, axis=1)
    assert_(eq(a2dma, [1.5, 4.0]))