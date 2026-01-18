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
@pytest.mark.parametrize('dtype', '?bhilqpBHILQPefdgFDGO')
def test_ones_pathological(self, dtype):
    arr = np.ones(10, dtype=dtype)
    expected = np.zeros(10, dtype=dtype)
    actual = np.clip(arr, 1, 0)
    if dtype == 'O':
        assert actual.tolist() == expected.tolist()
    else:
        assert_equal(actual, expected)