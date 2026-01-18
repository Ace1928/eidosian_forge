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
def test_single_item_array_non_int(self):
    res = delete(np.ones(1), np.array([False]))
    assert_array_equal(res, np.ones(1))
    x = np.ones((3, 1))
    false_mask = np.array([False], dtype=bool)
    true_mask = np.array([True], dtype=bool)
    res = delete(x, false_mask, axis=-1)
    assert_array_equal(res, x)
    res = delete(x, true_mask, axis=-1)
    assert_array_equal(res, x[:, :0])
    with pytest.raises(IndexError):
        delete(np.ones(2), np.array([0], dtype=object))
    with pytest.raises(IndexError):
        delete(np.ones(2), np.array([0], dtype='m8[ns]'))