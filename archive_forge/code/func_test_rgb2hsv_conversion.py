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
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_rgb2hsv_conversion(self, channel_axis):
    rgb = img_as_float(self.img_rgb)[::16, ::16]
    _rgb = np.moveaxis(rgb, source=-1, destination=channel_axis)
    hsv = rgb2hsv(_rgb, channel_axis=channel_axis)
    hsv = np.moveaxis(hsv, source=channel_axis, destination=-1)
    hsv = hsv.reshape(-1, 3)
    gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2]) for pt in rgb.reshape(-1, 3)])
    assert_almost_equal(hsv, gt)