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
def test_ynp_zeros(self):
    ao = special.ynp_zeros(0, 2)
    assert_array_almost_equal(ao, array([2.19714133, 5.42968104]), 6)
    ao = special.ynp_zeros(43, 5)
    assert_allclose(special.yvp(43, ao), 0, atol=1e-15)
    ao = special.ynp_zeros(443, 5)
    assert_allclose(special.yvp(443, ao), 0, atol=1e-09)