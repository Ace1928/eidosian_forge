import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_unpackbits():
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.array([[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1, 1]]))