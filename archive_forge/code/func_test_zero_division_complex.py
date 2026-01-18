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
def test_zero_division_complex(self):
    with np.errstate(invalid='ignore', divide='ignore'):
        x = np.array([0.0], dtype=np.complex128)
        y = 1.0 / x
        assert_(np.isinf(y)[0])
        y = complex(np.inf, np.nan) / x
        assert_(np.isinf(y)[0])
        y = complex(np.nan, np.inf) / x
        assert_(np.isinf(y)[0])
        y = complex(np.inf, np.inf) / x
        assert_(np.isinf(y)[0])
        y = 0.0 / x
        assert_(np.isnan(y)[0])