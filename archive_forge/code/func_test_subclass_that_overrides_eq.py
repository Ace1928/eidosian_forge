import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_subclass_that_overrides_eq(self):

    class MyArray(np.ndarray):

        def __eq__(self, other):
            return bool(np.equal(self, other).all())

        def __ne__(self, other):
            return not self == other
    a = np.array([1.0, 2.0]).view(MyArray)
    b = np.array([2.0, 3.0]).view(MyArray)
    assert_(type(a == a), bool)
    assert_(a == a)
    assert_(a != b)
    self._test_equal(a, a)
    self._test_not_equal(a, b)
    self._test_not_equal(b, a)