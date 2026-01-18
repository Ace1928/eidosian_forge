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
@pytest.mark.parametrize('method', quantile_methods)
def test_quantile_monotonic(self, method):
    p0 = np.linspace(0, 1, 101)
    quantile = np.quantile(np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7]) * 0.1, p0, method=method)
    assert_equal(np.sort(quantile), quantile)
    quantile = np.quantile([0.0, 1.0, 2.0, 3.0], p0, method=method)
    assert_equal(np.sort(quantile), quantile)