import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float16, np.float32])
def test_triangle_uniform_images(dtype):
    assert threshold_triangle(np.zeros((10, 10), dtype=dtype)) == 0
    assert threshold_triangle(np.ones((10, 10), dtype=dtype)) == 1
    assert threshold_triangle(np.full((10, 10), 2, dtype=dtype)) == 2