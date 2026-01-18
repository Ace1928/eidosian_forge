import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_ufunc_override_mro(self):

    def tres_mul(a, b, c):
        return a * b * c

    def quatro_mul(a, b, c, d):
        return a * b * c * d
    three_mul_ufunc = np.frompyfunc(tres_mul, 3, 1)
    four_mul_ufunc = np.frompyfunc(quatro_mul, 4, 1)

    class A:

        def __array_ufunc__(self, func, method, *inputs, **kwargs):
            return 'A'

    class ASub(A):

        def __array_ufunc__(self, func, method, *inputs, **kwargs):
            return 'ASub'

    class B:

        def __array_ufunc__(self, func, method, *inputs, **kwargs):
            return 'B'

    class C:

        def __init__(self):
            self.count = 0

        def __array_ufunc__(self, func, method, *inputs, **kwargs):
            self.count += 1
            return NotImplemented

    class CSub(C):

        def __array_ufunc__(self, func, method, *inputs, **kwargs):
            self.count += 1
            return NotImplemented
    a = A()
    a_sub = ASub()
    b = B()
    c = C()
    res = np.multiply(a, a_sub)
    assert_equal(res, 'ASub')
    res = np.multiply(a_sub, b)
    assert_equal(res, 'ASub')
    res = np.multiply(c, a)
    assert_equal(res, 'A')
    assert_equal(c.count, 1)
    res = np.multiply(c, a)
    assert_equal(c.count, 2)
    c = C()
    c_sub = CSub()
    assert_raises(TypeError, np.multiply, c, c_sub)
    assert_equal(c.count, 1)
    assert_equal(c_sub.count, 1)
    c.count = c_sub.count = 0
    assert_raises(TypeError, np.multiply, c_sub, c)
    assert_equal(c.count, 1)
    assert_equal(c_sub.count, 1)
    c.count = 0
    assert_raises(TypeError, np.multiply, c, c)
    assert_equal(c.count, 1)
    c.count = 0
    assert_raises(TypeError, np.multiply, 2, c)
    assert_equal(c.count, 1)
    assert_equal(three_mul_ufunc(a, 1, 2), 'A')
    assert_equal(three_mul_ufunc(1, a, 2), 'A')
    assert_equal(three_mul_ufunc(1, 2, a), 'A')
    assert_equal(three_mul_ufunc(a, a, 6), 'A')
    assert_equal(three_mul_ufunc(a, 2, a), 'A')
    assert_equal(three_mul_ufunc(a, 2, b), 'A')
    assert_equal(three_mul_ufunc(a, 2, a_sub), 'ASub')
    assert_equal(three_mul_ufunc(a, a_sub, 3), 'ASub')
    c.count = 0
    assert_equal(three_mul_ufunc(c, a_sub, 3), 'ASub')
    assert_equal(c.count, 1)
    c.count = 0
    assert_equal(three_mul_ufunc(1, a_sub, c), 'ASub')
    assert_equal(c.count, 0)
    c.count = 0
    assert_equal(three_mul_ufunc(a, b, c), 'A')
    assert_equal(c.count, 0)
    c_sub.count = 0
    assert_equal(three_mul_ufunc(a, b, c_sub), 'A')
    assert_equal(c_sub.count, 0)
    assert_equal(three_mul_ufunc(1, 2, b), 'B')
    assert_raises(TypeError, three_mul_ufunc, 1, 2, c)
    assert_raises(TypeError, three_mul_ufunc, c_sub, 2, c)
    assert_raises(TypeError, three_mul_ufunc, c_sub, 2, 3)
    assert_equal(four_mul_ufunc(a, 1, 2, 3), 'A')
    assert_equal(four_mul_ufunc(1, a, 2, 3), 'A')
    assert_equal(four_mul_ufunc(1, 1, a, 3), 'A')
    assert_equal(four_mul_ufunc(1, 1, 2, a), 'A')
    assert_equal(four_mul_ufunc(a, b, 2, 3), 'A')
    assert_equal(four_mul_ufunc(1, a, 2, b), 'A')
    assert_equal(four_mul_ufunc(b, 1, a, 3), 'B')
    assert_equal(four_mul_ufunc(a_sub, 1, 2, a), 'ASub')
    assert_equal(four_mul_ufunc(a, 1, 2, a_sub), 'ASub')
    c = C()
    c_sub = CSub()
    assert_raises(TypeError, four_mul_ufunc, 1, 2, 3, c)
    assert_equal(c.count, 1)
    c.count = 0
    assert_raises(TypeError, four_mul_ufunc, 1, 2, c_sub, c)
    assert_equal(c_sub.count, 1)
    assert_equal(c.count, 1)
    c2 = C()
    c.count = c_sub.count = 0
    assert_raises(TypeError, four_mul_ufunc, 1, c, c_sub, c2)
    assert_equal(c_sub.count, 1)
    assert_equal(c.count, 1)
    assert_equal(c2.count, 0)
    c.count = c2.count = c_sub.count = 0
    assert_raises(TypeError, four_mul_ufunc, c2, c, c_sub, c)
    assert_equal(c_sub.count, 1)
    assert_equal(c.count, 0)
    assert_equal(c2.count, 1)