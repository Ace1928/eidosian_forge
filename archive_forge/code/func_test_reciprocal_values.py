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
def test_reciprocal_values(self):
    with np.errstate(all='ignore'):
        x = [np.nan, np.nan, 0.0, -0.0, np.inf, -np.inf]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0]
        for dt in ['e', 'f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.reciprocal(yf), xf)
    with np.errstate(divide='raise'):
        for dt in ['e', 'f', 'd', 'g']:
            assert_raises(FloatingPointError, np.reciprocal, np.array(-0.0, dtype=dt))