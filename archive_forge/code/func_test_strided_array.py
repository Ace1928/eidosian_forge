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
def test_strided_array(self):
    arr1 = np.array([-4.0, 1.0, 10.0, 0.0, np.nan, -np.nan, np.inf, -np.inf])
    arr2 = np.array([-2.0, -1.0, np.nan, 1.0, 0.0, np.nan, 1.0, -3.0])
    mintrue = np.array([-4.0, -1.0, np.nan, 0.0, np.nan, np.nan, 1.0, -np.inf])
    out = np.ones(8)
    out_mintrue = np.array([-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0])
    assert_equal(np.minimum(arr1, arr2), mintrue)
    assert_equal(np.minimum(arr1[::2], arr2[::2]), mintrue[::2])
    assert_equal(np.minimum(arr1[:4], arr2[::2]), np.array([-4.0, np.nan, 0.0, 0.0]))
    assert_equal(np.minimum(arr1[::3], arr2[:3]), np.array([-4.0, -1.0, np.nan]))
    assert_equal(np.minimum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-4.0, 1.0, np.nan]))
    assert_equal(out, out_mintrue)