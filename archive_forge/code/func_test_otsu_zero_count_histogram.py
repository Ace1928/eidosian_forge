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
def test_otsu_zero_count_histogram():
    """Issue #5497.

    As the histogram returned by np.bincount starts with zero,
    it resulted in NaN-related issues.
    """
    x = np.array([1, 2])
    t1 = threshold_otsu(x)
    t2 = threshold_otsu(hist=np.bincount(x))
    assert t1 == t2