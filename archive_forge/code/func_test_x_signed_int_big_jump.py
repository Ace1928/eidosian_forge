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
@pytest.mark.parametrize('x_dtype', [np.int8, np.int16, np.int32, np.int64])
def test_x_signed_int_big_jump(self, x_dtype):
    minint = np.iinfo(x_dtype).min
    maxint = np.iinfo(x_dtype).max
    x = np.array([-1, maxint], dtype=x_dtype)
    f = np.array([minint // 2, 0])
    dfdx = gradient(f, x)
    assert_array_equal(dfdx, [0.5, 0.5])