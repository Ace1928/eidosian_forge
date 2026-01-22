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
class TestMaskedArrayArithmetic:

    def setup_method(self):
        x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0])
        a10 = 10.0
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = np.array([-0.5, 0.0, 0.5, 0.8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.err_status)

    def test_basic_arithmetic(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        a2d = array([[1, 2], [0, 4]])
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        assert_equal(a2d * a2d, a2d * a2dm)
        assert_equal(a2d + a2d, a2d + a2dm)
        assert_equal(a2d - a2d, a2d - a2dm)
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            assert_equal(-x, -xm)
            assert_equal(x + y, xm + ym)
            assert_equal(x - y, xm - ym)
            assert_equal(x * y, xm * ym)
            assert_equal(x / y, xm / ym)
            assert_equal(a10 + y, a10 + ym)
            assert_equal(a10 - y, a10 - ym)
            assert_equal(a10 * y, a10 * ym)
            assert_equal(a10 / y, a10 / ym)
            assert_equal(x + a10, xm + a10)
            assert_equal(x - a10, xm - a10)
            assert_equal(x * a10, xm * a10)
            assert_equal(x / a10, xm / a10)
            assert_equal(x ** 2, xm ** 2)
            assert_equal(abs(x) ** 2.5, abs(xm) ** 2.5)
            assert_equal(x ** y, xm ** ym)
            assert_equal(np.add(x, y), add(xm, ym))
            assert_equal(np.subtract(x, y), subtract(xm, ym))
            assert_equal(np.multiply(x, y), multiply(xm, ym))
            assert_equal(np.divide(x, y), divide(xm, ym))

    def test_divide_on_different_shapes(self):
        x = arange(6, dtype=float)
        x.shape = (2, 3)
        y = arange(3, dtype=float)
        z = x / y
        assert_equal(z, [[-1.0, 1.0, 1.0], [-1.0, 4.0, 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])
        z = x / y[None, :]
        assert_equal(z, [[-1.0, 1.0, 1.0], [-1.0, 4.0, 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])
        y = arange(2, dtype=float)
        z = x / y[:, None]
        assert_equal(z, [[-1.0, -1.0, -1.0], [3.0, 4.0, 5.0]])
        assert_equal(z.mask, [[1, 1, 1], [0, 0, 0]])

    def test_mixed_arithmetic(self):
        na = np.array([1])
        ma = array([1])
        assert_(isinstance(na + ma, MaskedArray))
        assert_(isinstance(ma + na, MaskedArray))

    def test_limits_arithmetic(self):
        tiny = np.finfo(float).tiny
        a = array([tiny, 1.0 / tiny, 0.0])
        assert_equal(getmaskarray(a / 2), [0, 0, 0])
        assert_equal(getmaskarray(2 / a), [1, 0, 1])

    def test_masked_singleton_arithmetic(self):
        xm = array(0, mask=1)
        assert_((1 / array(0)).mask)
        assert_((1 + xm).mask)
        assert_((-xm).mask)
        assert_(maximum(xm, xm).mask)
        assert_(minimum(xm, xm).mask)

    def test_masked_singleton_equality(self):
        a = array([1, 2, 3], mask=[1, 1, 0])
        assert_((a[0] == 0) is masked)
        assert_((a[0] != 0) is masked)
        assert_equal(a[-1] == 0, False)
        assert_equal(a[-1] != 0, True)

    def test_arithmetic_with_masked_singleton(self):
        x = masked_array([1, 2])
        y = x * masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])
        y = x[0] * masked
        assert_(y is masked)
        y = x + masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])

    def test_arithmetic_with_masked_singleton_on_1d_singleton(self):
        x = masked_array([1])
        y = x + masked
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True])

    def test_scalar_arithmetic(self):
        x = array(0, mask=0)
        assert_equal(x.filled().ctypes.data, x.ctypes.data)
        xm = array((0, 0)) / 0.0
        assert_equal(xm.shape, (2,))
        assert_equal(xm.mask, [1, 1])

    def test_basic_ufuncs(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        assert_equal(np.cos(x), cos(xm))
        assert_equal(np.cosh(x), cosh(xm))
        assert_equal(np.sin(x), sin(xm))
        assert_equal(np.sinh(x), sinh(xm))
        assert_equal(np.tan(x), tan(xm))
        assert_equal(np.tanh(x), tanh(xm))
        assert_equal(np.sqrt(abs(x)), sqrt(xm))
        assert_equal(np.log(abs(x)), log(xm))
        assert_equal(np.log10(abs(x)), log10(xm))
        assert_equal(np.exp(x), exp(xm))
        assert_equal(np.arcsin(z), arcsin(zm))
        assert_equal(np.arccos(z), arccos(zm))
        assert_equal(np.arctan(z), arctan(zm))
        assert_equal(np.arctan2(x, y), arctan2(xm, ym))
        assert_equal(np.absolute(x), absolute(xm))
        assert_equal(np.angle(x + 1j * y), angle(xm + 1j * ym))
        assert_equal(np.angle(x + 1j * y, deg=True), angle(xm + 1j * ym, deg=True))
        assert_equal(np.equal(x, y), equal(xm, ym))
        assert_equal(np.not_equal(x, y), not_equal(xm, ym))
        assert_equal(np.less(x, y), less(xm, ym))
        assert_equal(np.greater(x, y), greater(xm, ym))
        assert_equal(np.less_equal(x, y), less_equal(xm, ym))
        assert_equal(np.greater_equal(x, y), greater_equal(xm, ym))
        assert_equal(np.conjugate(x), conjugate(xm))

    def test_count_func(self):
        assert_equal(1, count(1))
        assert_equal(0, array(1, mask=[1]))
        ott = array([0.0, 1.0, 2.0, 3.0], mask=[1, 0, 0, 0])
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        ott = ott.reshape((2, 2))
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_equal([1, 2], res)
        assert_(getmask(res) is nomask)
        ott = array([0.0, 1.0, 2.0, 3.0])
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_(res.dtype.type is np.intp)
        assert_raises(np.AxisError, ott.count, axis=1)

    def test_count_on_python_builtins(self):
        assert_equal(3, count([1, 2, 3]))
        assert_equal(2, count((1, 2)))

    def test_minmax_func(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        xr = np.ravel(x)
        xmr = ravel(xm)
        assert_equal(max(xr), maximum.reduce(xmr))
        assert_equal(min(xr), minimum.reduce(xmr))
        assert_equal(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3])
        assert_equal(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9])
        x = arange(5)
        y = arange(5) - 2
        x[3] = masked
        y[0] = masked
        assert_equal(minimum(x, y), where(less(x, y), x, y))
        assert_equal(maximum(x, y), where(greater(x, y), x, y))
        assert_(minimum.reduce(x) == 0)
        assert_(maximum.reduce(x) == 4)
        x = arange(4).reshape(2, 2)
        x[-1, -1] = masked
        assert_equal(maximum.reduce(x, axis=None), 2)

    def test_minimummaximum_func(self):
        a = np.ones((2, 2))
        aminimum = minimum(a, a)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum(a, a))
        aminimum = minimum.outer(a, a)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum.outer(a, a))
        amaximum = maximum(a, a)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum(a, a))
        amaximum = maximum.outer(a, a)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum.outer(a, a))

    def test_minmax_reduce(self):
        a = array([1, 2, 3], mask=[False, False, False])
        b = np.maximum.reduce(a)
        assert_equal(b, 3)

    def test_minmax_funcs_with_output(self):
        mask = np.random.rand(12).round()
        xm = array(np.random.uniform(0, 10, 12), mask=mask)
        xm.shape = (3, 4)
        for funcname in ('min', 'max'):
            npfunc = getattr(np, funcname)
            mafunc = getattr(numpy.ma.core, funcname)
            nout = np.empty((4,), dtype=int)
            try:
                result = npfunc(xm, axis=0, out=nout)
            except MaskError:
                pass
            nout = np.empty((4,), dtype=float)
            result = npfunc(xm, axis=0, out=nout)
            assert_(result is nout)
            nout.fill(-999)
            result = mafunc(xm, axis=0, out=nout)
            assert_(result is nout)

    def test_minmax_methods(self):
        _, _, _, _, _, xm, _, _, _, _ = self.d
        xm.shape = (xm.size,)
        assert_equal(xm.max(), 10)
        assert_(xm[0].max() is masked)
        assert_(xm[0].max(0) is masked)
        assert_(xm[0].max(-1) is masked)
        assert_equal(xm.min(), -10.0)
        assert_(xm[0].min() is masked)
        assert_(xm[0].min(0) is masked)
        assert_(xm[0].min(-1) is masked)
        assert_equal(xm.ptp(), 20.0)
        assert_(xm[0].ptp() is masked)
        assert_(xm[0].ptp(0) is masked)
        assert_(xm[0].ptp(-1) is masked)
        x = array([1, 2, 3], mask=True)
        assert_(x.min() is masked)
        assert_(x.max() is masked)
        assert_(x.ptp() is masked)

    def test_minmax_dtypes(self):
        x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
        a10 = 10.0
        an10 = -10.0
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        xm = masked_array(x, mask=m1)
        xm.set_fill_value(1e+20)
        float_dtypes = [np.float16, np.float32, np.float64, np.longdouble, np.complex64, np.complex128, np.clongdouble]
        for float_dtype in float_dtypes:
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(), float_dtype(a10))
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(), float_dtype(an10))
        assert_equal(xm.min(), an10)
        assert_equal(xm.max(), a10)
        for float_dtype in float_dtypes[:4]:
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(), float_dtype(a10))
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(), float_dtype(an10))
        for float_dtype in float_dtypes[-3:]:
            ym = masked_array([1e+20 + 1j, 1e+20 - 2j, 1e+20 - 1j], mask=[0, 1, 0], dtype=float_dtype)
            assert_equal(ym.min(), float_dtype(1e+20 - 1j))
            assert_equal(ym.max(), float_dtype(1e+20 + 1j))
            zm = masked_array([np.inf + 2j, np.inf + 3j, -np.inf - 1j], mask=[0, 1, 0], dtype=float_dtype)
            assert_equal(zm.min(), float_dtype(-np.inf - 1j))
            assert_equal(zm.max(), float_dtype(np.inf + 2j))
            cmax = np.inf - 1j * np.finfo(np.float64).max
            assert masked_array([-cmax, 0], mask=[0, 1]).max() == -cmax
            assert masked_array([cmax, 0], mask=[0, 1]).min() == cmax

    def test_addsumprod(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        assert_equal(np.add.reduce(x), add.reduce(x))
        assert_equal(np.add.accumulate(x), add.accumulate(x))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(np.sum(x, axis=0), sum(x, axis=0))
        assert_equal(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0))
        assert_equal(np.sum(x, 0), sum(x, 0))
        assert_equal(np.prod(x, axis=0), product(x, axis=0))
        assert_equal(np.prod(x, 0), product(x, 0))
        assert_equal(np.prod(filled(xm, 1), axis=0), product(xm, axis=0))
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        if len(s) > 1:
            assert_equal(np.concatenate((x, y), 1), concatenate((xm, ym), 1))
            assert_equal(np.add.reduce(x, 1), add.reduce(x, 1))
            assert_equal(np.sum(x, 1), sum(x, 1))
            assert_equal(np.prod(x, 1), product(x, 1))

    def test_binops_d2D(self):
        a = array([[1.0], [2.0], [3.0]], mask=[[False], [True], [True]])
        b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        test = a * b
        control = array([[2.0, 3.0], [2.0, 2.0], [3.0, 3.0]], mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        test = b * a
        control = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        a = array([[1.0], [2.0], [3.0]])
        b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [0, 0], [0, 1]])
        test = a * b
        control = array([[2, 3], [8, 10], [18, 3]], mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        test = b * a
        control = array([[2, 3], [8, 10], [18, 7]], mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_domained_binops_d2D(self):
        a = array([[1.0], [2.0], [3.0]], mask=[[False], [True], [True]])
        b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        test = a / b
        control = array([[1.0 / 2.0, 1.0 / 3.0], [2.0, 2.0], [3.0, 3.0]], mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        test = b / a
        control = array([[2.0 / 1.0, 3.0 / 1.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        a = array([[1.0], [2.0], [3.0]])
        b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [0, 0], [0, 1]])
        test = a / b
        control = array([[1.0 / 2, 1.0 / 3], [2.0 / 4, 2.0 / 5], [3.0 / 6, 3]], mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
        test = b / a
        control = array([[2 / 1.0, 3 / 1.0], [4 / 2.0, 5 / 2.0], [6 / 3.0, 7]], mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_noshrinking(self):
        a = masked_array([1.0, 2.0, 3.0], mask=[False, False, False], shrink=False)
        b = a + 1
        assert_equal(b.mask, [0, 0, 0])
        a += 1
        assert_equal(a.mask, [0, 0, 0])
        b = a / 1.0
        assert_equal(b.mask, [0, 0, 0])
        a /= 1.0
        assert_equal(a.mask, [0, 0, 0])

    def test_ufunc_nomask(self):
        m = np.ma.array([1])
        assert_equal(np.true_divide(m, 5).mask.shape, ())

    def test_noshink_on_creation(self):
        a = np.ma.masked_values([1.0, 2.5, 3.1], 1.5, shrink=False)
        assert_equal(a.mask, [0, 0, 0])

    def test_mod(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        assert_equal(mod(x, y), mod(xm, ym))
        test = mod(ym, xm)
        assert_equal(test, np.mod(ym, xm))
        assert_equal(test.mask, mask_or(xm.mask, ym.mask))
        test = mod(xm, ym)
        assert_equal(test, np.mod(xm, ym))
        assert_equal(test.mask, mask_or(mask_or(xm.mask, ym.mask), ym == 0))

    def test_TakeTransposeInnerOuter(self):
        x = arange(24)
        y = np.arange(24)
        x[5:6] = masked
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert_equal(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1)))
        assert_equal(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1))
        assert_equal(np.inner(filled(x, 0), filled(y, 0)), inner(x, y))
        assert_equal(np.outer(filled(x, 0), filled(y, 0)), outer(x, y))
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked
        t = take(y, [0, 3, 4])
        assert_(t[0] == 'abc')
        assert_(t[1] == 2)
        assert_(t[2] == 3)

    def test_imag_real(self):
        xx = array([1 + 10j, 20 + 2j], mask=[1, 0])
        assert_equal(xx.imag, [10, 2])
        assert_equal(xx.imag.filled(), [1e+20, 2])
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        assert_equal(xx.real, [1, 20])
        assert_equal(xx.real.filled(), [1e+20, 20])
        assert_equal(xx.real.dtype, xx._data.real.dtype)

    def test_methods_with_output(self):
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked
        funclist = ('sum', 'prod', 'var', 'std', 'max', 'min', 'ptp', 'mean')
        for funcname in funclist:
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            output = np.empty(4, dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))
            output = empty(4, dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)
            assert_(output[0] is masked)

    def test_eq_on_structured(self):
        ndtype = [('A', int), ('B', int)]
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)
        test = a == a
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        test = a == a[0]
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        test = a == b
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = a[0] == b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = a == b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))], [(3, (3, 3)), (4, (4, 4))]], mask=[[(0, (1, 0)), (0, (0, 1))], [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        test = a[0, 0] == a
        assert_equal(test.data, [[True, False], [False, False]])
        assert_equal(test.mask, [[False, False], [False, True]])
        assert_(test.fill_value == True)

    def test_ne_on_structured(self):
        ndtype = [('A', int), ('B', int)]
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)
        test = a != a
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        test = a != a[0]
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        test = a != b
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = a[0] != b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = a != b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))], [(3, (3, 3)), (4, (4, 4))]], mask=[[(0, (1, 0)), (0, (0, 1))], [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        test = a[0, 0] != a
        assert_equal(test.data, [[False, True], [True, True]])
        assert_equal(test.mask, [[False, False], [False, True]])
        assert_(test.fill_value == True)

    def test_eq_ne_structured_with_non_masked(self):
        a = array([(1, 1), (2, 2), (3, 4)], mask=[(0, 1), (0, 0), (1, 1)], dtype='i4,i4')
        eq = a == a.data
        ne = a.data != a
        assert_(np.all(eq))
        assert_(not np.any(ne))
        expected_mask = a.mask == np.ones((), a.mask.dtype)
        assert_array_equal(eq.mask, expected_mask)
        assert_array_equal(ne.mask, expected_mask)
        assert_equal(eq.data, [True, True, False])
        assert_array_equal(eq.data, ~ne.data)

    def test_eq_ne_structured_extra(self):
        dt = np.dtype('i4,i4')
        for m1 in (mvoid((1, 2), mask=(0, 0), dtype=dt), mvoid((1, 2), mask=(0, 1), dtype=dt), mvoid((1, 2), mask=(1, 0), dtype=dt), mvoid((1, 2), mask=(1, 1), dtype=dt)):
            ma1 = m1.view(MaskedArray)
            r1 = ma1.view('2i4')
            for m2 in (np.array((1, 1), dtype=dt), mvoid((1, 1), dtype=dt), mvoid((1, 0), mask=(0, 1), dtype=dt), mvoid((3, 2), mask=(0, 1), dtype=dt)):
                ma2 = m2.view(MaskedArray)
                r2 = ma2.view('2i4')
                eq_expected = (r1 == r2).all()
                assert_equal(m1 == m2, eq_expected)
                assert_equal(m2 == m1, eq_expected)
                assert_equal(ma1 == m2, eq_expected)
                assert_equal(m1 == ma2, eq_expected)
                assert_equal(ma1 == ma2, eq_expected)
                el_by_el = [m1[name] == m2[name] for name in dt.names]
                assert_equal(array(el_by_el, dtype=bool).all(), eq_expected)
                ne_expected = (r1 != r2).any()
                assert_equal(m1 != m2, ne_expected)
                assert_equal(m2 != m1, ne_expected)
                assert_equal(ma1 != m2, ne_expected)
                assert_equal(m1 != ma2, ne_expected)
                assert_equal(ma1 != ma2, ne_expected)
                el_by_el = [m1[name] != m2[name] for name in dt.names]
                assert_equal(array(el_by_el, dtype=bool).any(), ne_expected)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_eq_for_strings(self, dt, fill):
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)
        test = a == a
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        test = a == a[0]
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)
        test = a == b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)
        test = a[0] == b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = b == a[0]
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_ne_for_strings(self, dt, fill):
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)
        test = a != a
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        test = a != a[0]
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)
        test = a != b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)
        test = a[0] != b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = b != a[0]
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_eq_for_numeric(self, dt1, dt2, fill):
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)
        test = a == a
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        test = a == a[0]
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = a == b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)
        test = a[0] == b
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = b == a[0]
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('op', [operator.eq, operator.lt])
    def test_eq_broadcast_with_unmasked(self, op):
        a = array([0, 1], mask=[0, 1])
        b = np.arange(10).reshape(5, 2)
        result = op(a, b)
        assert_(result.mask.shape == b.shape)
        assert_equal(result.mask, np.zeros(b.shape, bool) | a.mask)

    @pytest.mark.parametrize('op', [operator.eq, operator.gt])
    def test_comp_no_mask_not_broadcast(self, op):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = op(a, 3)
        assert_(not result.mask.shape)
        assert_(result.mask is nomask)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_ne_for_numeric(self, dt1, dt2, fill):
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)
        test = a != a
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        test = a != a[0]
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = a != b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)
        test = a[0] != b
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = b != a[0]
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    @pytest.mark.parametrize('op', [operator.le, operator.lt, operator.ge, operator.gt])
    def test_comparisons_for_numeric(self, op, dt1, dt2, fill):
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)
        test = op(a, a)
        assert_equal(test.data, op(a._data, a._data))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        test = op(a, a[0])
        assert_equal(test.data, op(a._data, a._data[0]))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = op(a, b)
        assert_equal(test.data, op(a._data, b._data))
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)
        test = op(a[0], b)
        assert_equal(test.data, op(a._data[0], b._data))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
        test = op(b, a[0])
        assert_equal(test.data, op(b._data, a._data[0]))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('op', [operator.le, operator.lt, operator.ge, operator.gt])
    @pytest.mark.parametrize('fill', [None, 'N/A'])
    def test_comparisons_strings(self, op, fill):
        ma1 = masked_array(['a', 'b', 'cde'], mask=[0, 1, 0], fill_value=fill)
        ma2 = masked_array(['cde', 'b', 'a'], mask=[0, 1, 0], fill_value=fill)
        assert_equal(op(ma1, ma2)._data, op(ma1._data, ma2._data))

    def test_eq_with_None(self):
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, 'Comparison to `None`')
            a = array([None, 1], mask=[0, 1])
            assert_equal(a == None, array([True, False], mask=[0, 1]))
            assert_equal(a.data == None, [True, False])
            assert_equal(a != None, array([False, True], mask=[0, 1]))
            a = array([None, 1], mask=False)
            assert_equal(a == None, [True, False])
            assert_equal(a != None, [False, True])
            a = array([None, 2], mask=True)
            assert_equal(a == None, array([False, True], mask=True))
            assert_equal(a != None, array([True, False], mask=True))
            a = masked
            assert_equal(a == None, masked)

    def test_eq_with_scalar(self):
        a = array(1)
        assert_equal(a == 1, True)
        assert_equal(a == 0, False)
        assert_equal(a != 1, False)
        assert_equal(a != 0, True)
        b = array(1, mask=True)
        assert_equal(b == 0, masked)
        assert_equal(b == 1, masked)
        assert_equal(b != 0, masked)
        assert_equal(b != 1, masked)

    def test_eq_different_dimensions(self):
        m1 = array([1, 1], mask=[0, 1])
        for m2 in (array([[0, 1], [1, 2]]), np.array([[0, 1], [1, 2]])):
            test = m1 == m2
            assert_equal(test.data, [[False, False], [True, False]])
            assert_equal(test.mask, [[False, True], [False, True]])

    def test_numpyarithmetic(self):
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        control = masked_array([np.nan, np.nan, 0, np.log(2), -1], mask=[1, 1, 0, 0, 1])
        test = log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])
        test = np.log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])