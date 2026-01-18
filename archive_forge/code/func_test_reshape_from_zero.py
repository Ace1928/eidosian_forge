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
def test_reshape_from_zero(self):
    A = np.zeros(0, dtype=[('a', np.float32)])
    Ar = np.resize(A, (2, 1))
    assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
    assert_equal(A.dtype, Ar.dtype)