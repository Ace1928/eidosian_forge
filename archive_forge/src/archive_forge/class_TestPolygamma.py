import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestPolygamma:

    def test_polygamma(self):
        poly2 = special.polygamma(2, 1)
        poly3 = special.polygamma(3, 1)
        assert_almost_equal(poly2, -2.4041138063, 10)
        assert_almost_equal(poly3, 6.4939394023, 10)
        x = [2, 3, 110000000000000.0]
        assert_almost_equal(special.polygamma(0, x), special.psi(x))
        n = [0, 1, 2]
        x = [0.5, 1.5, 2.5]
        expected = [-1.9635100260214238, 0.9348022005446793, -0.2362040516417274]
        assert_almost_equal(special.polygamma(n, x), expected)
        expected = np.vstack([expected] * 2)
        assert_almost_equal(special.polygamma(n, np.vstack([x] * 2)), expected)
        assert_almost_equal(special.polygamma(np.vstack([n] * 2), x), expected)