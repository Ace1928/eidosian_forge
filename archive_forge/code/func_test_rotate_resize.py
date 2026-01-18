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
def test_rotate_resize():
    x = np.zeros((10, 10), dtype=np.float64)
    x45 = rotate(x, 45, resize=False)
    assert x45.shape == (10, 10)
    x45 = rotate(x, 45, resize=True)
    assert x45.shape == (14, 14)