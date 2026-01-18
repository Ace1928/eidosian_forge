import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_rgba2rgb_dtypes(dtype):
    rgba = np.array([[[0, 0.5, 1, 0], [0, 0.5, 1, 1], [0, 0.5, 1, 0.5]]]).astype(dtype=dtype)
    rgb = rgba2rgb(rgba)
    float_dtype = _supported_float_type(rgba.dtype)
    assert rgb.dtype == float_dtype
    expected = np.array([[[1, 1, 1], [0, 0.5, 1], [0.5, 0.75, 1]]]).astype(float)
    assert rgb.shape == expected.shape
    assert_almost_equal(rgb, expected)