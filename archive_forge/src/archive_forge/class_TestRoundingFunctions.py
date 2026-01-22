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
class TestRoundingFunctions:

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

    def test_object_indirect(self):
        """ test implementations via __float__ """

        class C:

            def __float__(self):
                return -2.5
        arr = np.array([C(), C()])
        assert_equal(np.floor(arr), [-3, -3])
        assert_equal(np.ceil(arr), [-2, -2])
        with pytest.raises(TypeError):
            np.trunc(arr)

    def test_fraction(self):
        f = Fraction(-4, 3)
        assert_equal(np.floor(f), -2)
        assert_equal(np.ceil(f), -1)
        assert_equal(np.trunc(f), -1)