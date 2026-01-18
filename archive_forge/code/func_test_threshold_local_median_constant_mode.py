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
def test_threshold_local_median_constant_mode(self):
    out = threshold_local(self.image, 3, method='median', mode='constant', cval=20)
    expected = np.array([[20.0, 1.0, 3.0, 4.0, 20.0], [1.0, 1.0, 3.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 1.0, 2.0], [20.0, 5.0, 5.0, 2.0, 20.0]])
    assert_equal(expected, out)