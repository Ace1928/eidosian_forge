import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
def test_subpix_no_class():
    img = np.zeros((50, 50))
    subpix = corner_subpix(img, np.array([[25, 25]]))
    assert_array_equal(subpix[0], (np.nan, np.nan))
    img[25, 25] = 1e-10
    corner = peak_local_max(corner_harris(img), min_distance=10, threshold_rel=0, num_peaks=1)
    subpix = corner_subpix(img, corner)
    assert_array_equal(subpix[0], (np.nan, np.nan))