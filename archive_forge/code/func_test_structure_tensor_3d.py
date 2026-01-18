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
def test_structure_tensor_3d(dtype):
    cube = np.zeros((5, 5, 5), dtype=dtype)
    cube[2, 2, 2] = 1
    A_elems = structure_tensor(cube, sigma=0.1)
    assert all((a.dtype == _supported_float_type(dtype) for a in A_elems))
    assert_equal(len(A_elems), 6)
    assert_array_equal(A_elems[0][:, 1, :], np.array([[0, 0, 0, 0, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]]))
    assert_array_equal(A_elems[0][1], np.array([[0, 0, 0, 0, 0], [0, 1, 4, 1, 0], [0, 4, 16, 4, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]]))
    assert_array_equal(A_elems[3][2], np.array([[0, 0, 0, 0, 0], [0, 4, 16, 4, 0], [0, 0, 0, 0, 0], [0, 4, 16, 4, 0], [0, 0, 0, 0, 0]]))