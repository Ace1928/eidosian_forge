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
def test_priority_with_scalar(self):

    class A(np.ndarray):
        __array_priority__ = 10

        def __new__(cls):
            return np.asarray(1.0, 'float64').view(cls).copy()
    a = A()
    x = np.float64(1) * a
    assert_(isinstance(x, A))
    assert_array_equal(x, np.array(1))