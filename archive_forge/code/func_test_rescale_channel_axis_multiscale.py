import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.ndimage import map_coordinates
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color.colorconv import rgb2gray
from skimage.data import checkerboard, astronaut
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage.transform._warps import (
from skimage.transform._geometric import (
from skimage.util.dtype import img_as_float, _convert
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_rescale_channel_axis_multiscale(channel_axis):
    x = np.zeros((5, 5, 3), dtype=np.float64)
    x = np.moveaxis(x, -1, channel_axis)
    scaled = rescale(x, scale=(2, 1), order=0, channel_axis=channel_axis, anti_aliasing=False, mode='constant')
    scaled = np.moveaxis(scaled, channel_axis, -1)
    assert scaled.shape == (10, 5, 3)