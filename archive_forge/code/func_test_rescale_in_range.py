import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_rescale_in_range(dtype):
    image = np.array([51.0, 102.0, 153.0], dtype=dtype)
    out = exposure.rescale_intensity(image, in_range=(0, 255))
    assert_array_almost_equal(out, [0.2, 0.4, 0.6], decimal=4)
    assert out.dtype == image.dtype