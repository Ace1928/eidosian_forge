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
def test_writeback(self):
    X = np.array([1.1, 2.2])
    Y = np.array([3.3, 4.4])
    x, y = np.meshgrid(X, Y, sparse=False, copy=True)
    x[0, :] = 0
    assert_equal(x[0, :], 0)
    assert_equal(x[1, :], X)