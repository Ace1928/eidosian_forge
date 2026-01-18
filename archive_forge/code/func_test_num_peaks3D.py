import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_num_peaks3D(self):
    image = np.zeros((10, 10, 100))
    image[5, 5, ::5] = np.arange(20)
    peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=2)
    assert len(peaks_limited) == 2