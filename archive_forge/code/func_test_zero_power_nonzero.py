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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_zero_power_nonzero(self):
    zero = np.array([0.0 + 0j])
    cnan = np.array([complex(np.nan, np.nan)])

    def assert_complex_equal(x, y):
        assert_array_equal(x.real, y.real)
        assert_array_equal(x.imag, y.imag)
    assert_complex_equal(np.power(zero, 1 + 4j), zero)
    assert_complex_equal(np.power(zero, 2 - 3j), zero)
    assert_complex_equal(np.power(zero, 1 + 1j), zero)
    assert_complex_equal(np.power(zero, 1 + 0j), zero)
    assert_complex_equal(np.power(zero, 1 - 1j), zero)
    with pytest.warns(expected_warning=RuntimeWarning) as r:
        assert_complex_equal(np.power(zero, -1 + 1j), cnan)
        assert_complex_equal(np.power(zero, -2 - 3j), cnan)
        assert_complex_equal(np.power(zero, -7 + 0j), cnan)
        assert_complex_equal(np.power(zero, 0 + 1j), cnan)
        assert_complex_equal(np.power(zero, 0 - 1j), cnan)
    assert len(r) == 5