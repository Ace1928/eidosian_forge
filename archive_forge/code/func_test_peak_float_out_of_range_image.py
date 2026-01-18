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
def test_peak_float_out_of_range_image(dtype):
    im = np.array([10, 100], dtype=dtype)
    frequencies, bin_centers = exposure.histogram(im, nbins=90)
    assert bin_centers.dtype == dtype
    assert_array_equal(bin_centers, np.arange(10, 100) + 0.5)