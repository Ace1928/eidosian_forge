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
@pytest.mark.parametrize('order', [0, 1, 3])
def test_warp_clip_cval_is_nan(order):
    x = np.ones((15, 15), dtype=np.float64)
    x[5:-5, 5:-5] = 2
    outx = rotate(x, 45, order=order, cval=np.nan, resize=True, clip=True)
    assert_array_almost_equal(np.nanmin(outx), 1)
    assert_array_almost_equal(np.nanmax(outx), 2)