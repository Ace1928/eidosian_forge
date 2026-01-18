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
def test_nonzero_twodim(self):
    x = np.array([[0, 1, 0], [2, 0, 3]])
    assert_equal(np.count_nonzero(x.astype('i1')), 3)
    assert_equal(np.count_nonzero(x.astype('i2')), 3)
    assert_equal(np.count_nonzero(x.astype('i4')), 3)
    assert_equal(np.count_nonzero(x.astype('i8')), 3)
    assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))
    x = np.eye(3)
    assert_equal(np.count_nonzero(x.astype('i1')), 3)
    assert_equal(np.count_nonzero(x.astype('i2')), 3)
    assert_equal(np.count_nonzero(x.astype('i4')), 3)
    assert_equal(np.count_nonzero(x.astype('i8')), 3)
    assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))
    x = np.array([[(0, 1), (0, 0), (1, 11)], [(1, 1), (1, 0), (0, 0)], [(0, 0), (1, 5), (0, 1)]], dtype=[('a', 'f4'), ('b', 'u1')])
    assert_equal(np.count_nonzero(x['a']), 4)
    assert_equal(np.count_nonzero(x['b']), 5)
    assert_equal(np.nonzero(x['a']), ([0, 1, 1, 2], [2, 0, 1, 1]))
    assert_equal(np.nonzero(x['b']), ([0, 0, 1, 2, 2], [0, 2, 0, 1, 2]))
    assert_(not x['a'].T.flags.aligned)
    assert_equal(np.count_nonzero(x['a'].T), 4)
    assert_equal(np.count_nonzero(x['b'].T), 5)
    assert_equal(np.nonzero(x['a'].T), ([0, 1, 1, 2], [1, 1, 2, 0]))
    assert_equal(np.nonzero(x['b'].T), ([0, 0, 1, 2, 2], [0, 1, 2, 0, 2]))