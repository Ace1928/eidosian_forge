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
class TestPtp:

    def test_basic(self):
        a = np.array([3, 4, 5, 10, -3, -5, 6.0])
        assert_equal(a.ptp(axis=0), 15.0)
        b = np.array([[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]])
        assert_equal(b.ptp(axis=0), [5.0, 7.0, 7.0])
        assert_equal(b.ptp(axis=-1), [6.0, 6.0, 6.0])
        assert_equal(b.ptp(axis=0, keepdims=True), [[5.0, 7.0, 7.0]])
        assert_equal(b.ptp(axis=(0, 1), keepdims=True), [[8.0]])