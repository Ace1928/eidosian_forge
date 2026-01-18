import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_subclass_that_does_not_implement_npall(self):

    class MyArray(np.ndarray):

        def __array_function__(self, *args, **kwargs):
            return NotImplemented
    a = np.array([1.0, 2.0]).view(MyArray)
    b = np.array([2.0, 3.0]).view(MyArray)
    with assert_raises(TypeError):
        np.all(a)
    self._test_equal(a, a)
    self._test_not_equal(a, b)
    self._test_not_equal(b, a)