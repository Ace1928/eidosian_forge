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
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2, -3])
def test_resize_local_mean3d_keep(channel_axis):
    nch = 3
    x = np.zeros((5, 5, nch), dtype=np.float64)
    x[1, 1, :] = 1
    x = np.moveaxis(x, -1, channel_axis)
    resized = resize_local_mean(x, (10, 10), channel_axis=channel_axis)
    resized = np.moveaxis(resized, channel_axis, -1)
    with pytest.raises(ValueError):
        resize_local_mean(x, (10,))
    ref = np.zeros((10, 10, nch))
    ref[2:4, 2:4, :] = 1
    assert_array_almost_equal(resized, ref)
    channel_axis = channel_axis % x.ndim
    spatial_shape = (10, 10)
    out_shape = spatial_shape[:channel_axis] + (nch,) + spatial_shape[channel_axis:]
    resized = resize_local_mean(x, out_shape)
    resized = np.moveaxis(resized, channel_axis, -1)
    assert_array_almost_equal(resized, ref)