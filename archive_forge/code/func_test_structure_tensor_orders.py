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
def test_structure_tensor_orders():
    square = np.zeros((5, 5))
    square[2, 2] = 1
    A_elems_default = structure_tensor(square, sigma=0.1)
    A_elems_xy = structure_tensor(square, sigma=0.1, order='xy')
    A_elems_rc = structure_tensor(square, sigma=0.1, order='rc')
    assert_array_equal(A_elems_rc, A_elems_default)
    assert_array_equal(A_elems_xy, A_elems_default[::-1])