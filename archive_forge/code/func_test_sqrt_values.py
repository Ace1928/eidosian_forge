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
@pytest.mark.parametrize('dt', ['e', 'f', 'd', 'g'])
def test_sqrt_values(self, dt):
    with np.errstate(all='ignore'):
        x = [np.nan, np.nan, np.inf, np.nan, 0.0]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0.0]
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        assert_equal(np.sqrt(yf), xf)