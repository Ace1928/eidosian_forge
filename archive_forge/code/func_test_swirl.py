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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_swirl(dtype):
    image = img_as_float(checkerboard()).astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)
    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    with expected_warnings(['Bi-quadratic.*bug']):
        swirled = swirl(image, strength=10, **swirl_params)
        unswirled = swirl(swirled, strength=-10, **swirl_params)
        assert swirled.dtype == unswirled.dtype == float_dtype
    assert np.mean(np.abs(image - unswirled)) < 0.01
    swirl_params.pop('mode')
    with expected_warnings(['Bi-quadratic.*bug']):
        swirled = swirl(image, strength=10, **swirl_params)
        unswirled = swirl(swirled, strength=-10, **swirl_params)
        assert swirled.dtype == unswirled.dtype == float_dtype
    assert np.mean(np.abs(image[1:-1, 1:-1] - unswirled[1:-1, 1:-1])) < 0.01