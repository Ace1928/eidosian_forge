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
def test_rgb2luv_brucelindbloom(self):
    """
        Test the RGB->Lab conversion by comparing to the calculator on the
        authoritative Bruce Lindbloom
        [website](http://brucelindbloom.com/index.html?ColorCalculator.html).
        """
    gt_for_colbars = np.array([[100, 0, 0], [97.1393, 7.7056, 106.7866], [91.1132, -70.4773, -15.2042], [87.7347, -83.0776, 107.3985], [60.3242, 84.0714, -108.6834], [53.2408, 175.0151, 37.7564], [32.297, -9.4054, -130.3423], [0, 0, 0]]).T
    gt_array = np.swapaxes(gt_for_colbars.reshape(3, 4, 2), 0, 2)
    assert_array_almost_equal(rgb2luv(self.colbars_array), gt_array, decimal=2)