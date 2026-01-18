import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize('swap', ['', 'swap'])
@pytest.mark.parametrize('string_dtype', ['U', 'S'])
def test_promote_types_strings(self, swap, string_dtype):
    if swap == 'swap':
        promote_types = lambda a, b: np.promote_types(b, a)
    else:
        promote_types = np.promote_types
    S = string_dtype
    assert_equal(promote_types('bool', S), np.dtype(S + '5'))
    assert_equal(promote_types('b', S), np.dtype(S + '4'))
    assert_equal(promote_types('u1', S), np.dtype(S + '3'))
    assert_equal(promote_types('u2', S), np.dtype(S + '5'))
    assert_equal(promote_types('u4', S), np.dtype(S + '10'))
    assert_equal(promote_types('u8', S), np.dtype(S + '20'))
    assert_equal(promote_types('i1', S), np.dtype(S + '4'))
    assert_equal(promote_types('i2', S), np.dtype(S + '6'))
    assert_equal(promote_types('i4', S), np.dtype(S + '11'))
    assert_equal(promote_types('i8', S), np.dtype(S + '21'))
    assert_equal(promote_types('bool', S + '1'), np.dtype(S + '5'))
    assert_equal(promote_types('bool', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('b', S + '1'), np.dtype(S + '4'))
    assert_equal(promote_types('b', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('u1', S + '1'), np.dtype(S + '3'))
    assert_equal(promote_types('u1', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('u2', S + '1'), np.dtype(S + '5'))
    assert_equal(promote_types('u2', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('u4', S + '1'), np.dtype(S + '10'))
    assert_equal(promote_types('u4', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('u8', S + '1'), np.dtype(S + '20'))
    assert_equal(promote_types('u8', S + '30'), np.dtype(S + '30'))
    assert_equal(promote_types('O', S + '30'), np.dtype('O'))