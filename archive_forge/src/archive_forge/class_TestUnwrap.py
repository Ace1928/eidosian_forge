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
class TestUnwrap:

    def test_simple(self):
        assert_array_equal(unwrap([1, 1 + 2 * np.pi]), [1, 1])
        assert_(np.all(diff(unwrap(rand(10) * 100)) < np.pi))

    def test_period(self):
        assert_array_equal(unwrap([1, 1 + 256], period=255), [1, 2])
        assert_(np.all(diff(unwrap(rand(10) * 1000, period=255)) < 255))
        simple_seq = np.array([0, 75, 150, 225, 300])
        wrap_seq = np.mod(simple_seq, 255)
        assert_array_equal(unwrap(wrap_seq, period=255), simple_seq)
        uneven_seq = np.array([0, 75, 150, 225, 300, 430])
        wrap_uneven = np.mod(uneven_seq, 250)
        no_discont = unwrap(wrap_uneven, period=250)
        assert_array_equal(no_discont, [0, 75, 150, 225, 300, 180])
        sm_discont = unwrap(wrap_uneven, period=250, discont=140)
        assert_array_equal(sm_discont, [0, 75, 150, 225, 300, 430])
        assert sm_discont.dtype == wrap_uneven.dtype