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
def test_truth_table_bitwise(self):
    arg1 = [False, False, True, True]
    arg2 = [False, True, False, True]
    out = [False, True, True, True]
    assert_equal(np.bitwise_or(arg1, arg2), out)
    out = [False, False, False, True]
    assert_equal(np.bitwise_and(arg1, arg2), out)
    out = [False, True, True, False]
    assert_equal(np.bitwise_xor(arg1, arg2), out)