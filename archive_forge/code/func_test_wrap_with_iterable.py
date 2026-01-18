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
def test_wrap_with_iterable(self):

    class with_wrap(np.ndarray):
        __array_priority__ = 10

        def __new__(cls):
            return np.asarray(1).view(cls).copy()

        def __array_wrap__(self, arr, context):
            return arr.view(type(self))
    a = with_wrap()
    x = ncu.multiply(a, (1, 2, 3))
    assert_(isinstance(x, with_wrap))
    assert_array_equal(x, np.array((1, 2, 3)))