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
def test_yuv_roundtrip(self, channel_axis):
    img_rgb = img_as_float(self.img_rgb)[::16, ::16]
    img_rgb = np.moveaxis(img_rgb, source=-1, destination=channel_axis)
    assert_array_almost_equal(yuv2rgb(rgb2yuv(img_rgb, channel_axis=channel_axis), channel_axis=channel_axis), img_rgb)
    assert_array_almost_equal(yiq2rgb(rgb2yiq(img_rgb, channel_axis=channel_axis), channel_axis=channel_axis), img_rgb)
    assert_array_almost_equal(ypbpr2rgb(rgb2ypbpr(img_rgb, channel_axis=channel_axis), channel_axis=channel_axis), img_rgb)
    assert_array_almost_equal(ycbcr2rgb(rgb2ycbcr(img_rgb, channel_axis=channel_axis), channel_axis=channel_axis), img_rgb)
    assert_array_almost_equal(ydbdr2rgb(rgb2ydbdr(img_rgb, channel_axis=channel_axis), channel_axis=channel_axis), img_rgb)