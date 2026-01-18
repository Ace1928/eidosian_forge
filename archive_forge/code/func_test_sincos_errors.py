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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('callable', [np.sin, np.cos])
@pytest.mark.parametrize('dtype', ['e', 'f', 'd'])
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_sincos_errors(self, callable, dtype, value):
    with np.errstate(invalid='raise'):
        assert_raises(FloatingPointError, callable, np.array([value], dtype=dtype))