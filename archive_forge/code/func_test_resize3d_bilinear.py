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
def test_resize3d_bilinear():
    x = np.zeros((5, 5, 2), dtype=np.float64)
    x[1, 1, 0] = 0
    x[1, 1, 1] = 1
    resized = resize(x, (10, 10, 1), order=1, mode='constant', anti_aliasing=False)
    ref = np.zeros((10, 10, 1))
    ref[1:5, 1:5, :] = 0.03125
    ref[1:5, 2:4, :] = 0.09375
    ref[2:4, 1:5, :] = 0.09375
    ref[2:4, 2:4, :] = 0.28125
    assert_array_almost_equal(resized, ref)