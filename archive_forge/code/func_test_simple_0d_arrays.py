import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_simple_0d_arrays(self):
    x = np.array(1234.22)
    y = np.array(1234.23)
    self._assert_func(x, y, significant=5)
    self._assert_func(x, y, significant=6)
    assert_raises(AssertionError, lambda: self._assert_func(x, y, significant=7))