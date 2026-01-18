import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_right_left_behavior(self):
    for size in range(1, 10):
        xp = np.arange(size, dtype=np.double)
        yp = np.ones(size, dtype=np.double)
        incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
        decpts = incpts[::-1]
        incres = interp(incpts, xp, yp)
        decres = interp(decpts, xp, yp)
        inctgt = np.array([1, 1, 1, 1], dtype=float)
        dectgt = inctgt[::-1]
        assert_equal(incres, inctgt)
        assert_equal(decres, dectgt)
        incres = interp(incpts, xp, yp, left=0)
        decres = interp(decpts, xp, yp, left=0)
        inctgt = np.array([0, 1, 1, 1], dtype=float)
        dectgt = inctgt[::-1]
        assert_equal(incres, inctgt)
        assert_equal(decres, dectgt)
        incres = interp(incpts, xp, yp, right=2)
        decres = interp(decpts, xp, yp, right=2)
        inctgt = np.array([1, 1, 1, 2], dtype=float)
        dectgt = inctgt[::-1]
        assert_equal(incres, inctgt)
        assert_equal(decres, dectgt)
        incres = interp(incpts, xp, yp, left=0, right=2)
        decres = interp(decpts, xp, yp, left=0, right=2)
        inctgt = np.array([0, 1, 1, 2], dtype=float)
        dectgt = inctgt[::-1]
        assert_equal(incres, inctgt)
        assert_equal(decres, dectgt)