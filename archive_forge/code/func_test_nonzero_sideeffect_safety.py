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
def test_nonzero_sideeffect_safety(self):

    class FalseThenTrue:
        _val = False

        def __bool__(self):
            try:
                return self._val
            finally:
                self._val = True

    class TrueThenFalse:
        _val = True

        def __bool__(self):
            try:
                return self._val
            finally:
                self._val = False
    a = np.array([True, FalseThenTrue()])
    assert_raises(RuntimeError, np.nonzero, a)
    a = np.array([[True], [FalseThenTrue()]])
    assert_raises(RuntimeError, np.nonzero, a)
    a = np.array([False, TrueThenFalse()])
    assert_raises(RuntimeError, np.nonzero, a)
    a = np.array([[False], [TrueThenFalse()]])
    assert_raises(RuntimeError, np.nonzero, a)