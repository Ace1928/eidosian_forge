import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_num_peaks_tot_vs_labels_4quadrants(self):
    np.random.seed(21)
    image = np.random.uniform(size=(20, 30))
    i, j = np.mgrid[0:20, 0:30]
    labels = 1 + (i >= 10) + (j >= 15) * 2
    result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=np.inf, num_peaks_per_label=2)
    assert len(result) == 8
    result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=np.inf, num_peaks_per_label=1)
    assert len(result) == 4
    result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=2, num_peaks_per_label=2)
    assert len(result) == 2