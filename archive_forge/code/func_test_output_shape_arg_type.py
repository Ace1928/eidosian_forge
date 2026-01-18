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
@pytest.mark.parametrize('_type', [tuple, np.asarray, list])
def test_output_shape_arg_type(_type):
    img = np.random.rand(3, 3)
    output_shape = _type([5, 5])
    assert resize(img, output_shape).shape == tuple(output_shape)