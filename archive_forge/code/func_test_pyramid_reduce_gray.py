import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
def test_pyramid_reduce_gray():
    rows, cols = image_gray.shape
    out1 = pyramids.pyramid_reduce(image_gray, downscale=2, channel_axis=None)
    assert_array_equal(out1.shape, (rows / 2, cols / 2))
    assert_almost_equal(np.ptp(out1), 1.0, decimal=2)
    out2 = pyramids.pyramid_reduce(image_gray, downscale=2, channel_axis=None, preserve_range=True)
    assert_almost_equal(np.ptp(out2) / np.ptp(image_gray), 1.0, decimal=2)