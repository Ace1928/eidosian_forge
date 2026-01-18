import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_pyramid_expand_rgb(channel_axis):
    image = data.astronaut()
    rows, cols, dim = image.shape
    image = np.moveaxis(image, source=-1, destination=channel_axis)
    out = pyramids.pyramid_expand(image, upscale=2, channel_axis=channel_axis)
    expected_shape = [rows * 2, cols * 2]
    expected_shape.insert(channel_axis % image.ndim, dim)
    assert_array_equal(out.shape, expected_shape)