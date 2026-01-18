import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_reduce_axis_float_index(self):
    d = np.zeros((3, 3, 3))
    assert_raises(TypeError, np.min, d, 0.5)
    assert_raises(TypeError, np.min, d, (0.5, 1))
    assert_raises(TypeError, np.min, d, (1, 2.2))
    assert_raises(TypeError, np.min, d, (0.2, 1.2))