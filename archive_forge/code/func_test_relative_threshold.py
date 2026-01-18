import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_relative_threshold(self):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 10
    image[3, 3] = 20
    peaks = peak.peak_local_max(image, min_distance=1, threshold_rel=0.5)
    assert len(peaks) == 1
    assert_array_almost_equal(peaks, [(3, 3)])