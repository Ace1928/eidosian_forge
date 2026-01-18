import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_void_scalar_empty_tuple(self):
    s = np.zeros((), dtype='V4')
    assert_equal(s[()].dtype, s.dtype)
    assert_equal(s[()], s)
    assert_equal(type(s[...]), np.ndarray)