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
def test_power_zero(self):
    zero = np.array([0j])
    one = np.array([1 + 0j])
    cnan = np.array([complex(np.nan, np.nan)])

    def assert_complex_equal(x, y):
        x, y = (np.asarray(x), np.asarray(y))
        assert_array_equal(x.real, y.real)
        assert_array_equal(x.imag, y.imag)
    for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
        assert_complex_equal(np.power(zero, p), zero)
    assert_complex_equal(np.power(zero, 0), one)
    with np.errstate(invalid='ignore'):
        assert_complex_equal(np.power(zero, 0 + 1j), cnan)
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(np.power(zero, -p), cnan)
        assert_complex_equal(np.power(zero, -1 + 0.2j), cnan)