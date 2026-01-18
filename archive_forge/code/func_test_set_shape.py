import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_set_shape():
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, 6]])
    numpy_arr.shape = (6,)
    modin_arr = np.array([[1, 2, 3], [4, 5, 6]])
    modin_arr.shape = (6,)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    modin_arr.shape = 6
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    with pytest.raises(ValueError, match='cannot reshape'):
        modin_arr.shape = (4,)