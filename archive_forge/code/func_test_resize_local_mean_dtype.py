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
def test_resize_local_mean_dtype():
    x = np.zeros((5, 5))
    x_f32 = x.astype(np.float32)
    x_u8 = x.astype(np.uint8)
    x_b = x.astype(bool)
    assert resize_local_mean(x, (10, 10), preserve_range=False).dtype == x.dtype
    assert resize_local_mean(x, (10, 10), preserve_range=True).dtype == x.dtype
    assert resize_local_mean(x_u8, (10, 10), preserve_range=False).dtype == np.float64
    assert resize_local_mean(x_u8, (10, 10), preserve_range=True).dtype == np.float64
    assert resize_local_mean(x_b, (10, 10), preserve_range=False).dtype == np.float64
    assert resize_local_mean(x_b, (10, 10), preserve_range=True).dtype == np.float64
    assert resize_local_mean(x_f32, (10, 10), preserve_range=False).dtype == x_f32.dtype
    assert resize_local_mean(x_f32, (10, 10), preserve_range=True).dtype == x_f32.dtype