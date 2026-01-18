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
def test_second_order_accurate(self):
    x = np.linspace(0, 1, 10)
    dx = x[1] - x[0]
    y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
    analytical = 6 * x ** 2 + 8 * x + 2
    num_error = np.abs(np.gradient(y, dx, edge_order=2) / analytical - 1)
    assert_(np.all(num_error < 0.03) == True)
    np.random.seed(0)
    x = np.sort(np.random.random(10))
    y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
    analytical = 6 * x ** 2 + 8 * x + 2
    num_error = np.abs(np.gradient(y, x, edge_order=2) / analytical - 1)
    assert_(np.all(num_error < 0.03) == True)