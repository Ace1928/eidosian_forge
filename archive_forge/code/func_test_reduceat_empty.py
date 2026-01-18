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
def test_reduceat_empty():
    """Reduceat should work with empty arrays"""
    indices = np.array([], 'i4')
    x = np.array([], 'f8')
    result = np.add.reduceat(x, indices)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0,))
    x = np.ones((5, 2))
    result = np.add.reduceat(x, [], axis=0)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0, 2))
    result = np.add.reduceat(x, [], axis=1)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (5, 0))