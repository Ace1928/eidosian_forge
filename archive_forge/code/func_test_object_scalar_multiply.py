import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_object_scalar_multiply():
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.multiply(arr, 3), desired)
    assert_equal(np.multiply(3, arr), desired)