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
def _test_spacing(t):
    one = t(1)
    eps = np.finfo(t).eps
    nan = t(np.nan)
    inf = t(np.inf)
    with np.errstate(invalid='ignore'):
        assert_equal(np.spacing(one), eps)
        assert_(np.isnan(np.spacing(nan)))
        assert_(np.isnan(np.spacing(inf)))
        assert_(np.isnan(np.spacing(-inf)))
        assert_(np.spacing(t(1e+30)) != 0)