import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
class TestNQuad:

    def test_fixed_limits(self):

        def func1(x0, x1, x2, x3):
            val = x0 ** 2 + x1 * x2 - x3 ** 3 + np.sin(x0) + (1 if x0 - 0.2 * x3 - 0.5 - 0.25 * x1 > 0 else 0)
            return val

        def opts_basic(*args):
            return {'points': [0.2 * args[2] + 0.5 + 0.25 * args[0]]}
        res = nquad(func1, [[0, 1], [-1, 1], [0.13, 0.8], [-0.15, 1]], opts=[opts_basic, {}, {}, {}], full_output=True)
        assert_quad(res[:-1], 1.5267454070738635)
        assert_(res[-1]['neval'] > 0 and res[-1]['neval'] < 400000.0)

    def test_variable_limits(self):
        scale = 0.1

        def func2(x0, x1, x2, x3, t0, t1):
            val = x0 * x1 * x3 ** 2 + np.sin(x2) + 1 + (1 if x0 + t1 * x1 - t0 > 0 else 0)
            return val

        def lim0(x1, x2, x3, t0, t1):
            return [scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) - 1, scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) + 1]

        def lim1(x2, x3, t0, t1):
            return [scale * (t0 * x2 + t1 * x3) - 1, scale * (t0 * x2 + t1 * x3) + 1]

        def lim2(x3, t0, t1):
            return [scale * (x3 + t0 ** 2 * t1 ** 3) - 1, scale * (x3 + t0 ** 2 * t1 ** 3) + 1]

        def lim3(t0, t1):
            return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]

        def opts0(x1, x2, x3, t0, t1):
            return {'points': [t0 - t1 * x1]}

        def opts1(x2, x3, t0, t1):
            return {}

        def opts2(x3, t0, t1):
            return {}

        def opts3(t0, t1):
            return {}
        res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0), opts=[opts0, opts1, opts2, opts3])
        assert_quad(res, 25.066666666666663)

    def test_square_separate_ranges_and_opts(self):

        def f(y, x):
            return 1.0
        assert_quad(nquad(f, [[-1, 1], [-1, 1]], opts=[{}, {}]), 4.0)

    def test_square_aliased_ranges_and_opts(self):

        def f(y, x):
            return 1.0
        r = [-1, 1]
        opt = {}
        assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)

    def test_square_separate_fn_ranges_and_opts(self):

        def f(y, x):
            return 1.0

        def fn_range0(*args):
            return (-1, 1)

        def fn_range1(*args):
            return (-1, 1)

        def fn_opt0(*args):
            return {}

        def fn_opt1(*args):
            return {}
        ranges = [fn_range0, fn_range1]
        opts = [fn_opt0, fn_opt1]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_square_aliased_fn_ranges_and_opts(self):

        def f(y, x):
            return 1.0

        def fn_range(*args):
            return (-1, 1)

        def fn_opt(*args):
            return {}
        ranges = [fn_range, fn_range]
        opts = [fn_opt, fn_opt]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_matching_quad(self):

        def func(x):
            return x ** 2 + 1
        res, reserr = quad(func, 0, 4)
        res2, reserr2 = nquad(func, ranges=[[0, 4]])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_dblquad(self):

        def func2d(x0, x1):
            return x0 ** 2 + x1 ** 3 - x0 * x1 + 1
        res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
        res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_tplquad(self):

        def func3d(x0, x1, x2, c0, c1):
            return x0 ** 2 + c0 * x1 ** 3 - x0 * x1 + 1 + c1 * np.sin(x2)
        res = tplquad(func3d, -1, 2, lambda x: -2, lambda x: 2, lambda x, y: -np.pi, lambda x, y: np.pi, args=(2, 3))
        res2 = nquad(func3d, [[-np.pi, np.pi], [-2, 2], (-1, 2)], args=(2, 3))
        assert_almost_equal(res, res2)

    def test_dict_as_opts(self):
        try:
            nquad(lambda x, y: x * y, [[0, 1], [0, 1]], opts={'epsrel': 0.0001})
        except TypeError:
            assert False