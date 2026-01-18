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
@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('preserve_range', [True, False])
@pytest.mark.parametrize('anti_aliasing', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.uint8])
def test_resize_clip(order, preserve_range, anti_aliasing, dtype):
    if dtype == np.uint8 and (preserve_range or order == 0):
        expected_max = 255
    else:
        expected_max = 1.0
    x = np.ones((5, 5), dtype=dtype)
    if dtype == np.uint8:
        x *= 255
    else:
        x[0, 0] = np.nan
    resized = resize(x, (3, 3), order=order, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    assert np.nanmax(resized) == expected_max