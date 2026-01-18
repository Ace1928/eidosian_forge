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
def test_peak_int_range_dtype():
    im = np.array([10, 100], dtype=np.int8)
    frequencies, bin_centers = exposure.histogram(im, source_range='dtype')
    assert_array_equal(bin_centers, np.arange(-128, 128))
    assert_equal(frequencies[128 + 10], 1)
    assert_equal(frequencies[128 + 100], 1)
    assert_equal(frequencies[128 + 101], 0)
    assert_equal(frequencies.shape, (256,))