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
@pytest.mark.parametrize('dtype', ['e', 'f', 'd', 'g'])
def test_sincos_values(self, dtype):
    with np.errstate(all='ignore'):
        x = [np.nan, np.nan, np.nan, np.nan]
        y = [np.nan, -np.nan, np.inf, -np.inf]
        xf = np.array(x, dtype=dtype)
        yf = np.array(y, dtype=dtype)
        assert_equal(np.sin(yf), xf)
        assert_equal(np.cos(yf), xf)