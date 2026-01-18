import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_single_bool_index(self):
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_equal(a[np.array(True)], a[None])
    assert_equal(a[np.array(False)], a[None][0:0])