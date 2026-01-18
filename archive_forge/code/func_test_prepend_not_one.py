import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_prepend_not_one(self):
    assign = self.assign
    s_ = np.s_
    a = np.zeros(5)
    assert_raises(ValueError, assign, a, s_[...], np.ones((2, 1)))
    assert_raises(ValueError, assign, a, s_[[1, 2, 3],], np.ones((2, 1)))
    assert_raises(ValueError, assign, a, s_[[[1], [2]],], np.ones((2, 2, 1)))