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
def test_reduce_reorder(self):
    for n in (2, 4, 8, 16, 32):
        for dt in (np.float32, np.float16, np.complex64):
            for r in np.diagflat(np.array([np.nan] * n, dtype=dt)):
                assert_equal(np.min(r), np.nan)