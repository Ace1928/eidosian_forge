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
class TestRollaxis:
    tgtshape = {(0, 0): (1, 2, 3, 4), (0, 1): (1, 2, 3, 4), (0, 2): (2, 1, 3, 4), (0, 3): (2, 3, 1, 4), (0, 4): (2, 3, 4, 1), (1, 0): (2, 1, 3, 4), (1, 1): (1, 2, 3, 4), (1, 2): (1, 2, 3, 4), (1, 3): (1, 3, 2, 4), (1, 4): (1, 3, 4, 2), (2, 0): (3, 1, 2, 4), (2, 1): (1, 3, 2, 4), (2, 2): (1, 2, 3, 4), (2, 3): (1, 2, 3, 4), (2, 4): (1, 2, 4, 3), (3, 0): (4, 1, 2, 3), (3, 1): (1, 4, 2, 3), (3, 2): (1, 2, 4, 3), (3, 3): (1, 2, 3, 4), (3, 4): (1, 2, 3, 4)}

    def test_exceptions(self):
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        assert_raises(np.AxisError, np.rollaxis, a, -5, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, -5)
        assert_raises(np.AxisError, np.rollaxis, a, 4, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, 5)

    def test_results(self):
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        aind = np.indices(a.shape)
        assert_(a.flags['OWNDATA'])
        for i, j in self.tgtshape:
            res = np.rollaxis(a, axis=i, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[i, j], str((i, j)))
            assert_(not res.flags['OWNDATA'])
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[4 - ip, j])
            assert_(not res.flags['OWNDATA'])
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[i, 4 - jp])
            assert_(not res.flags['OWNDATA'])
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[4 - ip, 4 - jp])
            assert_(not res.flags['OWNDATA'])