import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
import pytest
from scipy import ndimage
@pytest.mark.xfail(True, reason='Broken on many platforms')
def test_uint64_max():
    big = 2 ** 64 - 1025
    arr = np.array([big, big, big], dtype=np.uint64)
    inds = np.indices(arr.shape) - 0.1
    x = ndimage.map_coordinates(arr, inds)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))
    x = ndimage.shift(arr, 0.1)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))