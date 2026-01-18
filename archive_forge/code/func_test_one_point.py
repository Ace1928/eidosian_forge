import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_one_point(self):
    image = np.zeros((10, 20))
    labels = np.zeros((10, 20), int)
    image[5, 5] = 1
    labels[5, 5] = 1
    peak_idx = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
    result = np.zeros_like(image, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert np.all(result == (labels == 1))