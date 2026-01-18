from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyint(self):
    assert_raises(TypeError, poly.polyint, [0], 0.5)
    assert_raises(ValueError, poly.polyint, [0], -1)
    assert_raises(ValueError, poly.polyint, [0], 1, [0, 0])
    assert_raises(ValueError, poly.polyint, [0], lbnd=[0])
    assert_raises(ValueError, poly.polyint, [0], scl=[0])
    assert_raises(TypeError, poly.polyint, [0], axis=0.5)
    with assert_warns(DeprecationWarning):
        poly.polyint([1, 1], 1.0)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = poly.polyint([0], m=i, k=k)
        assert_almost_equal(res, [0, 1])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        res = poly.polyint(pol, m=1, k=[i])
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        res = poly.polyint(pol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(poly.polyval(-1, res), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        res = poly.polyint(pol, m=1, k=[i], scl=2)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = poly.polyint(tgt, m=1)
            res = poly.polyint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = poly.polyint(tgt, m=1, k=[k])
            res = poly.polyint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = poly.polyint(tgt, m=1, k=[k], lbnd=-1)
            res = poly.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = poly.polyint(tgt, m=1, k=[k], scl=2)
            res = poly.polyint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))