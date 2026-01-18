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
def test_warp_nd():
    for dim in range(2, 8):
        shape = dim * (5,)
        x = np.zeros(shape, dtype=np.float64)
        x_c = dim * (2,)
        x[x_c] = 1
        refx = np.zeros(shape, dtype=np.float64)
        refx_c = dim * (1,)
        refx[refx_c] = 1
        coord_grid = dim * (slice(0, 5, 1),)
        coords = np.array(np.mgrid[coord_grid]) + 1
        outx = warp(x, coords, order=0, cval=0)
        assert_array_almost_equal(outx, refx)