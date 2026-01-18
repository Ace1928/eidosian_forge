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
@hypothesis.given(arr=arrays(dtype=np.float64, shape=st.integers(min_value=3, max_value=1000), elements=st.floats(allow_infinity=False, allow_nan=False, min_value=-1e+300, max_value=1e+300)))
def test_quantile_monotonic_hypo(self, arr):
    p0 = np.arange(0, 1, 0.01)
    quantile = np.quantile(arr, p0)
    assert_equal(np.sort(quantile), quantile)