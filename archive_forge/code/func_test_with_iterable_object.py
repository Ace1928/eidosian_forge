import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_with_iterable_object(self):
    d = np.array([[{1, 11}, {2, 22}, {3, 33}], [{4, 44}, {5, 55}, {6, 66}]])
    actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
    expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])
    assert_equal(actual, expected)
    for i in np.ndindex(actual.shape):
        assert_equal(type(actual[i]), type(expected[i]))