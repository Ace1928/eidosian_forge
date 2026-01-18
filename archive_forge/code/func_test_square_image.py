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
@run_in_parallel()
def test_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.0
    results = corner_moravec(im) > 0
    assert np.count_nonzero(results) == 92
    results = peak_local_max(corner_harris(im, method='k'), min_distance=10, threshold_rel=0)
    assert len(results) == 1
    results = peak_local_max(corner_harris(im, method='eps'), min_distance=10, threshold_rel=0)
    assert len(results) == 1
    results = peak_local_max(corner_shi_tomasi(im), min_distance=10, threshold_rel=0)
    assert len(results) == 1