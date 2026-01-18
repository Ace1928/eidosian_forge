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
def test_ufunc_override_disabled(self):

    class OptOut:
        __array_ufunc__ = None
    opt_out = OptOut()
    msg = "operand 'OptOut' does not support ufuncs"
    with assert_raises_regex(TypeError, msg):
        np.add(opt_out, 1)
    with assert_raises_regex(TypeError, msg):
        np.add(1, opt_out)
    with assert_raises_regex(TypeError, msg):
        np.negative(opt_out)

    class GreedyArray:

        def __array_ufunc__(self, *args, **kwargs):
            return self
    greedy = GreedyArray()
    assert_(np.negative(greedy) is greedy)
    with assert_raises_regex(TypeError, msg):
        np.add(greedy, opt_out)
    with assert_raises_regex(TypeError, msg):
        np.add(greedy, 1, out=opt_out)