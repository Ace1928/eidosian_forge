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
def test_nonzero_onedim(self):
    x = np.array([1, 0, 2, -1, 0, 0, 8])
    assert_equal(np.count_nonzero(x), 4)
    assert_equal(np.count_nonzero(x), 4)
    assert_equal(np.nonzero(x), ([0, 2, 3, 6],))
    x = np.array([(1, 2, -5, -3), (0, 0, 2, 7), (1, 1, 0, 1), (-1, 3, 1, 0), (0, 7, 0, 4)], dtype=[('a', 'i4'), ('b', 'i2'), ('c', 'i1'), ('d', 'i8')])
    assert_equal(np.count_nonzero(x['a']), 3)
    assert_equal(np.count_nonzero(x['b']), 4)
    assert_equal(np.count_nonzero(x['c']), 3)
    assert_equal(np.count_nonzero(x['d']), 4)
    assert_equal(np.nonzero(x['a']), ([0, 2, 3],))
    assert_equal(np.nonzero(x['b']), ([0, 2, 3, 4],))