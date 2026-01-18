import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_packbits():
    a = [[[1, 0, 1], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]]
    for dt in '?bBhHiIlLqQ':
        arr = np.array(a, dtype=dt)
        b = np.packbits(arr, axis=-1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, np.array([[[160], [64]], [[192], [32]]]))
    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))