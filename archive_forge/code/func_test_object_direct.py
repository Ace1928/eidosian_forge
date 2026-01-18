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
def test_object_direct(self):
    """ test direct implementation of these magic methods """

    class C:

        def __floor__(self):
            return 1

        def __ceil__(self):
            return 2

        def __trunc__(self):
            return 3
    arr = np.array([C(), C()])
    assert_equal(np.floor(arr), [1, 1])
    assert_equal(np.ceil(arr), [2, 2])
    assert_equal(np.trunc(arr), [3, 3])