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
def test_rescale_same_values():
    image = np.ones((2, 2))
    out = exposure.rescale_intensity(image)
    assert ~np.isnan(out).all()
    assert_array_almost_equal(out, image)