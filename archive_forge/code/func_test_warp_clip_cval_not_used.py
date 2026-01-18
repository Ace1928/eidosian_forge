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
@pytest.mark.parametrize('order', range(6))
def test_warp_clip_cval_not_used(order):
    x = np.ones((15, 15), dtype=np.float64)
    x[5:-5, 5:-5] = 2
    transform = AffineTransform(scale=15 / (15 + 2), translation=(1, 1))
    with expected_warnings(['Bi-quadratic.*bug'] if order == 2 else None):
        outx = warp(x, transform, mode='constant', order=order, cval=0, clip=True)
    assert_array_almost_equal(outx.min(), 1)