import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_simple_arrays(self):
    x = np.array([1.1, 2.2])
    y = np.array([1.2, 2.3])
    self._assert_func(x, y)
    assert_raises(AssertionError, lambda: self._assert_func(y, x))
    y = np.array([1.0, 2.3])
    assert_raises(AssertionError, lambda: self._assert_func(x, y))
    assert_raises(AssertionError, lambda: self._assert_func(y, x))