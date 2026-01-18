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
def test_structure_tensor_eigenvalues_3d():
    image = np.pad(cube(9, dtype=np.int64), 5, mode='constant') * 1000
    boundary = (np.pad(cube(9), 5, mode='constant') - np.pad(cube(7), 6, mode='constant')).astype(bool)
    A_elems = structure_tensor(image, sigma=0.1)
    e0, e1, e2 = structure_tensor_eigenvalues(A_elems)
    assert np.all(e0[boundary] != 0)