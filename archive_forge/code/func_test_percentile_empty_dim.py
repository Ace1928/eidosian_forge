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
def test_percentile_empty_dim(self):
    d = np.arange(11 * 2).reshape(11, 1, 2, 1)
    assert_array_equal(np.percentile(d, 50, axis=0).shape, (1, 2, 1))
    assert_array_equal(np.percentile(d, 50, axis=1).shape, (11, 2, 1))
    assert_array_equal(np.percentile(d, 50, axis=2).shape, (11, 1, 1))
    assert_array_equal(np.percentile(d, 50, axis=3).shape, (11, 1, 2))
    assert_array_equal(np.percentile(d, 50, axis=-1).shape, (11, 1, 2))
    assert_array_equal(np.percentile(d, 50, axis=-2).shape, (11, 1, 1))
    assert_array_equal(np.percentile(d, 50, axis=-3).shape, (11, 2, 1))
    assert_array_equal(np.percentile(d, 50, axis=-4).shape, (1, 2, 1))
    assert_array_equal(np.percentile(d, 50, axis=2, method='midpoint').shape, (11, 1, 1))
    assert_array_equal(np.percentile(d, 50, axis=-2, method='midpoint').shape, (11, 1, 1))
    assert_array_equal(np.array(np.percentile(d, [10, 50], axis=0)).shape, (2, 1, 2, 1))
    assert_array_equal(np.array(np.percentile(d, [10, 50], axis=1)).shape, (2, 11, 2, 1))
    assert_array_equal(np.array(np.percentile(d, [10, 50], axis=2)).shape, (2, 11, 1, 1))
    assert_array_equal(np.array(np.percentile(d, [10, 50], axis=3)).shape, (2, 11, 1, 2))