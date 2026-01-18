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
def test_structure_tensor_3d_rc_only():
    cube = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        structure_tensor(cube, sigma=0.1, order='xy')
    A_elems_rc = structure_tensor(cube, sigma=0.1, order='rc')
    A_elems_none = structure_tensor(cube, sigma=0.1)
    assert_array_equal(A_elems_rc, A_elems_none)