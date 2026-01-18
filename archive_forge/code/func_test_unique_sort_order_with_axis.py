import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_sort_order_with_axis(self):
    fmt = "sort order incorrect for integer type '%s'"
    for dt in 'bhilq':
        a = np.array([[-1], [0]], dt)
        b = np.unique(a, axis=0)
        assert_array_equal(a, b, fmt % dt)