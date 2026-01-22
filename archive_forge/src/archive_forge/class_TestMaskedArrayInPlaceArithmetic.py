import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedArrayInPlaceArithmetic:

    def setup_method(self):
        x = arange(10)
        y = arange(10)
        xm = arange(10)
        xm[2] = masked
        self.intdata = (x, y, xm)
        self.floatdata = (x.astype(float), y.astype(float), xm.astype(float))
        self.othertypes = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        self.othertypes = [np.dtype(_).type for _ in self.othertypes]
        self.uint8data = (x.astype(np.uint8), y.astype(np.uint8), xm.astype(np.uint8))

    def test_inplace_addition_scalar(self):
        x, y, xm = self.intdata
        xm[2] = masked
        x += 1
        assert_equal(x, y + 1)
        xm += 1
        assert_equal(xm, y + 1)
        x, _, xm = self.floatdata
        id1 = x.data.ctypes.data
        x += 1.0
        assert_(id1 == x.data.ctypes.data)
        assert_equal(x, y + 1.0)

    def test_inplace_addition_array(self):
        x, y, xm = self.intdata
        m = xm.mask
        a = arange(10, dtype=np.int16)
        a[-1] = masked
        x += a
        xm += a
        assert_equal(x, y + a)
        assert_equal(xm, y + a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar(self):
        x, y, xm = self.intdata
        x -= 1
        assert_equal(x, y - 1)
        xm -= 1
        assert_equal(xm, y - 1)

    def test_inplace_subtraction_array(self):
        x, y, xm = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x -= a
        xm -= a
        assert_equal(x, y - a)
        assert_equal(xm, y - a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_multiplication_scalar(self):
        x, y, xm = self.floatdata
        x *= 2.0
        assert_equal(x, y * 2)
        xm *= 2.0
        assert_equal(xm, y * 2)

    def test_inplace_multiplication_array(self):
        x, y, xm = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x *= a
        xm *= a
        assert_equal(x, y * a)
        assert_equal(xm, y * a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_division_scalar_int(self):
        x, y, xm = self.intdata
        x = arange(10) * 2
        xm = arange(10) * 2
        xm[2] = masked
        x //= 2
        assert_equal(x, y)
        xm //= 2
        assert_equal(xm, y)

    def test_inplace_division_scalar_float(self):
        x, y, xm = self.floatdata
        x /= 2.0
        assert_equal(x, y / 2.0)
        xm /= arange(10)
        assert_equal(xm, ones((10,)))

    def test_inplace_division_array_float(self):
        x, y, xm = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x /= a
        xm /= a
        assert_equal(x, y / a)
        assert_equal(xm, y / a)
        assert_equal(xm.mask, mask_or(mask_or(m, a.mask), a == 0))

    def test_inplace_division_misc(self):
        x = [1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0]
        y = [5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0]
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = xm / ym
        assert_equal(z._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data, [1.0, 1.0, 1.0, -1.0, -pi / 2.0, 4.0, 5.0, 1.0, 1.0, 1.0, 2.0, 3.0])
        xm = xm.copy()
        xm /= ym
        assert_equal(xm._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data, [1.0, 1.0, 1.0, -1.0, -pi / 2.0, 4.0, 5.0, 1.0, 1.0, 1.0, 2.0, 3.0])

    def test_datafriendly_add(self):
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x + 1
        assert_equal(xx.data, [2, 3, 3])
        assert_equal(xx.mask, [0, 0, 1])
        x += 1
        assert_equal(x.data, [2, 3, 3])
        assert_equal(x.mask, [0, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x + array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 4, 3])
        assert_equal(xx.mask, [1, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        x += array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 4, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_sub(self):
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - 1
        assert_equal(xx.data, [0, 1, 3])
        assert_equal(xx.mask, [0, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= 1
        assert_equal(x.data, [0, 1, 3])
        assert_equal(x.mask, [0, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 0, 3])
        assert_equal(xx.mask, [1, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 0, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_mul(self):
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * 2
        assert_equal(xx.data, [2, 4, 3])
        assert_equal(xx.mask, [0, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= 2
        assert_equal(x.data, [2, 4, 3])
        assert_equal(x.mask, [0, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 40, 3])
        assert_equal(xx.mask, [1, 0, 1])
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(x.data, [1, 40, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_div(self):
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x / 2.0
        assert_equal(xx.data, [1 / 2.0, 2 / 2.0, 3])
        assert_equal(xx.mask, [0, 0, 1])
        x = array([1.0, 2.0, 3.0], mask=[0, 0, 1])
        x /= 2.0
        assert_equal(x.data, [1 / 2.0, 2 / 2.0, 3])
        assert_equal(x.mask, [0, 0, 1])
        x = array([1.0, 2.0, 3.0], mask=[0, 0, 1])
        xx = x / array([10.0, 20.0, 30.0], mask=[1, 0, 0])
        assert_equal(xx.data, [1.0, 2.0 / 20.0, 3.0])
        assert_equal(xx.mask, [1, 0, 1])
        x = array([1.0, 2.0, 3.0], mask=[0, 0, 1])
        x /= array([10.0, 20.0, 30.0], mask=[1, 0, 0])
        assert_equal(x.data, [1.0, 2 / 20.0, 3.0])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_pow(self):
        x = array([1.0, 2.0, 3.0], mask=[0, 0, 1])
        xx = x ** 2.5
        assert_equal(xx.data, [1.0, 2.0 ** 2.5, 3.0])
        assert_equal(xx.mask, [0, 0, 1])
        x **= 2.5
        assert_equal(x.data, [1.0, 2.0 ** 2.5, 3])
        assert_equal(x.mask, [0, 0, 1])

    def test_datafriendly_add_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_sub_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_mul_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_inplace_addition_scalar_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                xm[2] = masked
                x += t(1)
                assert_equal(x, y + t(1))
                xm += t(1)
                assert_equal(xm, y + t(1))

    def test_inplace_addition_array_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x += a
                xm += a
                assert_equal(x, y + a)
                assert_equal(xm, y + a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                x -= t(1)
                assert_equal(x, y - t(1))
                xm -= t(1)
                assert_equal(xm, y - t(1))

    def test_inplace_subtraction_array_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x -= a
                xm -= a
                assert_equal(x, y - a)
                assert_equal(xm, y - a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_multiplication_scalar_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                x *= t(2)
                assert_equal(x, y * t(2))
                xm *= t(2)
                assert_equal(xm, y * t(2))

    def test_inplace_multiplication_array_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x *= a
                xm *= a
                assert_equal(x, y * a)
                assert_equal(xm, y * a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_floor_division_scalar_type(self):
        unsupported = {np.dtype(t).type for t in np.typecodes['Complex']}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                x = arange(10, dtype=t) * t(2)
                xm = arange(10, dtype=t) * t(2)
                xm[2] = masked
                try:
                    x //= t(2)
                    xm //= t(2)
                    assert_equal(x, y)
                    assert_equal(xm, y)
                except TypeError:
                    msg = f'Supported type {t} throwing TypeError'
                    assert t in unsupported, msg

    def test_inplace_floor_division_array_type(self):
        unsupported = {np.dtype(t).type for t in np.typecodes['Complex']}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                try:
                    x //= a
                    xm //= a
                    assert_equal(x, y // a)
                    assert_equal(xm, y // a)
                    assert_equal(xm.mask, mask_or(mask_or(m, a.mask), a == t(0)))
                except TypeError:
                    msg = f'Supported type {t} throwing TypeError'
                    assert t in unsupported, msg

    def test_inplace_division_scalar_type(self):
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                x = arange(10, dtype=t) * t(2)
                xm = arange(10, dtype=t) * t(2)
                xm[2] = masked
                try:
                    x /= t(2)
                    assert_equal(x, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= t(2)
                    assert_equal(xm, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')

    def test_inplace_division_array_type(self):
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                x, y, xm = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(xm.mask, mask_or(mask_or(m, a.mask), a == t(0)))
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')

    def test_inplace_pow_type(self):
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                x = array([1, 2, 3], mask=[0, 0, 1], dtype=t)
                xx = x ** t(2)
                xx_r = array([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
                assert_equal(xx.data, xx_r.data)
                assert_equal(xx.mask, xx_r.mask)
                x **= t(2)
                assert_equal(x.data, xx_r.data)
                assert_equal(x.mask, xx_r.mask)