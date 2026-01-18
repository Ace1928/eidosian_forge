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
def test_period(self):
    x = [-180, -170, -185, 185, -10, -5, 0, 365]
    xp = [190, -190, 350, -350]
    fp = [5, 10, 3, 4]
    y = [7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75]
    assert_almost_equal(np.interp(x, xp, fp, period=360), y)
    x = np.array(x, order='F').reshape(2, -1)
    y = np.array(y, order='C').reshape(2, -1)
    assert_almost_equal(np.interp(x, xp, fp, period=360), y)