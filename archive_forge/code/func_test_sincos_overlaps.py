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
@pytest.mark.parametrize('callable', [np.sin, np.cos])
@pytest.mark.parametrize('dtype', ['f', 'd'])
@pytest.mark.parametrize('stride', [-1, 1, 2, 4, 5])
def test_sincos_overlaps(self, callable, dtype, stride):
    N = 100
    M = N // abs(stride)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N, dtype)
    y = callable(x[::stride])
    callable(x[::stride], out=x[:M])
    assert_equal(x[:M], y)