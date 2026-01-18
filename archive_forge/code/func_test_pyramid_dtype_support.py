import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'uint8', 'int64'])
@pytest.mark.parametrize('pyramid_func', [pyramids.pyramid_gaussian, pyramids.pyramid_laplacian])
def test_pyramid_dtype_support(pyramid_func, dtype):
    img = np.random.randn(32, 8).astype(dtype)
    pyramid = pyramid_func(img)
    float_dtype = _supported_float_type(dtype)
    assert np.all([im.dtype == float_dtype for im in pyramid])