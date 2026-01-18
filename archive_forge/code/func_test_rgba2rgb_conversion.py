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
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_rgba2rgb_conversion(self, channel_axis):
    rgba = self.img_rgba
    rgba = np.moveaxis(rgba, source=-1, destination=channel_axis)
    rgb = rgba2rgb(rgba, channel_axis=channel_axis)
    rgb = np.moveaxis(rgb, source=channel_axis, destination=-1)
    expected = np.array([[[1, 1, 1], [0, 0.5, 1], [0.5, 0.75, 1]]]).astype(float)
    assert_equal(rgb.shape, expected.shape)
    assert_almost_equal(rgb, expected)