import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_num_peaks_and_labels(self):
    image = np.zeros((7, 7), dtype=np.uint8)
    labels = np.zeros((7, 7), dtype=np.uint8) + 20
    image[1, 1] = 10
    image[1, 3] = 11
    image[1, 5] = 12
    image[3, 5] = 8
    image[5, 3] = 7
    peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, labels=labels)
    assert len(peaks_limited) == 5
    peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, labels=labels, num_peaks=2)
    assert len(peaks_limited) == 2