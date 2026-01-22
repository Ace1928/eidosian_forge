import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestTrapz:

    def test_simple(self):
        x = np.arange(-10, 10, 0.1)
        r = trapz(np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
        assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)
        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2
        q = x[:, None, None] + y[None, :, None] + z[None, None, :]
        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)
        r = trapz(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)
        r = trapz(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z, axis=2)
        assert_almost_equal(r, qz)

    def test_masked(self):
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0
        assert_almost_equal(trapz(ym, x), r)
        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapz(ym, xm), r)
        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapz(y, xm), r)