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
def test_size_zero_output(self):
    f = np.vectorize(lambda x: x)
    x = np.zeros([0, 5], dtype=int)
    with assert_raises_regex(ValueError, 'otypes'):
        f(x)
    f.otypes = 'i'
    assert_array_equal(f(x), x)
    f = np.vectorize(lambda x: x, signature='()->()')
    with assert_raises_regex(ValueError, 'otypes'):
        f(x)
    f = np.vectorize(lambda x: x, signature='()->()', otypes='i')
    assert_array_equal(f(x), x)
    f = np.vectorize(lambda x: x, signature='(n)->(n)', otypes='i')
    assert_array_equal(f(x), x)
    f = np.vectorize(lambda x: x, signature='(n)->(n)')
    assert_array_equal(f(x.T), x.T)
    f = np.vectorize(lambda x: [x], signature='()->(n)', otypes='i')
    with assert_raises_regex(ValueError, 'new output dimensions'):
        f(x)