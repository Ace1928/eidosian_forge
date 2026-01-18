import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_partition_matrix_none():
    a = np.matrix([[2, 1, 0]])
    actual = np.partition(a, 1, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)