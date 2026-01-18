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
def test_place(self):
    assert_raises(TypeError, place, [1, 2, 3], [True, False], [0, 1])
    a = np.array([1, 4, 3, 2, 5, 8, 7])
    place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
    assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])
    place(a, np.zeros(7), [])
    assert_array_equal(a, np.arange(1, 8))
    place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
    assert_array_equal(a, [8, 2, 9, 4, 8, 6, 9])
    assert_raises_regex(ValueError, 'Cannot insert from an empty array', lambda: place(a, [0, 0, 0, 0, 0, 1, 0], []))
    a = np.array(['12', '34'])
    place(a, [0, 1], '9')
    assert_array_equal(a, ['12', '9'])