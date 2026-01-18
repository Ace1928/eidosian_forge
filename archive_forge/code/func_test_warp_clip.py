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
def test_warp_clip():
    x = np.zeros((5, 5), dtype=np.float64)
    x[2, 2] = 1
    outx = rescale(x, 3, order=3, clip=False, anti_aliasing=False, mode='constant')
    assert outx.min() < 0
    outx = rescale(x, 3, order=3, clip=True, anti_aliasing=False, mode='constant')
    assert_array_almost_equal(outx.min(), 0)
    assert_array_almost_equal(outx.max(), 1)