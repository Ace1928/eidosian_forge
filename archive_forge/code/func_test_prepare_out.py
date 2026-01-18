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
@pytest.mark.parametrize('use_where', [True, False])
def test_prepare_out(self, use_where):

    class with_prepare(np.ndarray):
        __array_priority__ = 10

        def __array_prepare__(self, arr, context):
            return np.array(arr).view(type=with_prepare)
    a = np.array([1]).view(type=with_prepare)
    if use_where:
        x = np.add(a, a, a, where=[True])
    else:
        x = np.add(a, a, a)
    assert_(not np.shares_memory(x, a))
    assert_equal(x, np.array([2]))
    assert_equal(type(x), with_prepare)