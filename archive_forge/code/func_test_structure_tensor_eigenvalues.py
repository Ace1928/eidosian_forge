import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_structure_tensor_eigenvalues(dtype):
    square = np.zeros((5, 5), dtype=dtype)
    square[2, 2] = 1
    A_elems = structure_tensor(square, sigma=0.1, order='rc')
    l1, l2 = structure_tensor_eigenvalues(A_elems)
    out_dtype = _supported_float_type(dtype)
    assert all((a.dtype == out_dtype for a in (l1, l2)))
    assert_array_equal(l1, np.array([[0, 0, 0, 0, 0], [0, 2, 4, 2, 0], [0, 4, 0, 4, 0], [0, 2, 4, 2, 0], [0, 0, 0, 0, 0]]))
    assert_array_equal(l2, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))