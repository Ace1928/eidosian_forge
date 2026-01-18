from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('arraylike', arraylikes())
def test_unpack_first_level(self, arraylike):
    obj = np.array([None])
    obj[0] = np.array(1.2)
    length = len(str(obj[0]))
    expected = np.dtype(f'S{length}')
    obj = arraylike(obj)
    arr = np.array([obj], dtype='S')
    assert arr.shape == (1, 1)
    assert arr.dtype == expected