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
def test_timedelta64(self):
    x = np.array([-5, -3, 10, 12, 61, 321, 300], dtype='timedelta64[D]')
    dx = np.array([2, 7, 7, 25, 154, 119, -21], dtype='timedelta64[D]')
    assert_array_equal(gradient(x), dx)
    assert_(dx.dtype == np.dtype('timedelta64[D]'))