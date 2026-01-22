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
class TestStdVarComplex:

    def test_basic(self):
        A = np.array([1, 1j, -1, -1j])
        real_var = 1
        assert_almost_equal(np.var(A), real_var)
        assert_almost_equal(np.std(A) ** 2, real_var)

    def test_scalars(self):
        assert_equal(np.var(1j), 0)
        assert_equal(np.std(1j), 0)