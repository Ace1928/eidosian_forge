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
def test_resize_nd():
    for dim in range(1, 6):
        shape = 2 + np.arange(dim) * 2
        x = np.ones(shape)
        out_shape = np.asarray(shape) * 1.5
        resized = resize(x, out_shape, order=0, mode='reflect', anti_aliasing=False)
        expected_shape = 1.5 * shape
        assert_array_equal(resized.shape, expected_shape)
        assert np.all(resized == 1)