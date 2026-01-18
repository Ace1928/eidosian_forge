import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_nanequals(self):
    a = np.array([1, 1, np.nan, np.nan, np.nan])
    unq = np.unique(a)
    not_unq = np.unique(a, equal_nan=False)
    assert_array_equal(unq, np.array([1, np.nan]))
    assert_array_equal(not_unq, np.array([1, np.nan, np.nan, np.nan]))