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
class TestHypot:

    def test_simple(self):
        assert_almost_equal(ncu.hypot(1, 1), ncu.sqrt(2))
        assert_almost_equal(ncu.hypot(0, 0), 0)

    def test_reduce(self):
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0, 0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([9.0, 12.0, 20.0]), 25.0)
        assert_equal(ncu.hypot.reduce([]), 0.0)