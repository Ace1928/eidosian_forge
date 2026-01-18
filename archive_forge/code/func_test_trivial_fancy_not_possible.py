import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_trivial_fancy_not_possible(self):
    a = np.arange(6)
    idx = np.arange(6, dtype=np.intp).reshape(2, 1, 3)[:, :, 0]
    assert_array_equal(a[idx], idx)
    a[idx] = -1
    res = np.arange(6)
    res[0] = -1
    res[3] = -1
    assert_array_equal(a, res)