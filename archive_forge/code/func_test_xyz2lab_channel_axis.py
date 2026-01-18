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
def test_xyz2lab_channel_axis(self, channel_axis):
    xyz = np.moveaxis(self.xyz_array, source=-1, destination=channel_axis)
    lab = xyz2lab(xyz, channel_axis=channel_axis)
    lab = np.moveaxis(lab, source=channel_axis, destination=-1)
    assert_array_almost_equal(lab, self.lab_array, decimal=3)