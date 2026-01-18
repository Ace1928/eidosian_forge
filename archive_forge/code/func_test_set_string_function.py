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
def test_set_string_function(self):
    a = np.array([1])
    np.set_string_function(lambda x: 'FOO', repr=True)
    assert_equal(repr(a), 'FOO')
    np.set_string_function(None, repr=True)
    assert_equal(repr(a), 'array([1])')
    np.set_string_function(lambda x: 'FOO', repr=False)
    assert_equal(str(a), 'FOO')
    np.set_string_function(None, repr=False)
    assert_equal(str(a), '[1]')