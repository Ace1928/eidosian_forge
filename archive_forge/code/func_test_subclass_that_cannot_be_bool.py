import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_subclass_that_cannot_be_bool(self):

    class MyArray(np.ndarray):

        def __eq__(self, other):
            return super().__eq__(other).view(np.ndarray)

        def __lt__(self, other):
            return super().__lt__(other).view(np.ndarray)

        def all(self, *args, **kwargs):
            raise NotImplementedError
    a = np.array([1.0, 2.0]).view(MyArray)
    self._assert_func(a, a)