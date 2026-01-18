from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='Needs 64bit platform')
def test_too_large_array_error_paths(self):
    """Test the error paths, including for memory leaks"""
    arr = np.array(0, dtype='uint8')
    arr = np.broadcast_to(arr, 2 ** 62)
    for i in range(5):
        with pytest.raises(MemoryError):
            np.array(arr)
        with pytest.raises(MemoryError):
            np.array([arr])