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
def test_shape_index():
    square = np.zeros((5, 5))
    square[2, 2] = 4
    with expected_warnings(['divide by zero|\\A\\Z', 'invalid value|\\A\\Z']):
        s = shape_index(square, sigma=0.1)
    assert_almost_equal(s, np.array([[np.nan, np.nan, -0.5, np.nan, np.nan], [np.nan, 0, np.nan, 0, np.nan], [-0.5, np.nan, -1, np.nan, -0.5], [np.nan, 0, np.nan, 0, np.nan], [np.nan, np.nan, -0.5, np.nan, np.nan]]))