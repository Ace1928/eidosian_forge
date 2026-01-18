import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_threshold_rel_default(self):
    image = np.ones((5, 5))
    image[2, 2] = 1
    assert len(peak.peak_local_max(image)) == 0
    image[2, 2] = 2
    assert_array_equal(peak.peak_local_max(image), [[2, 2]])
    image[2, 2] = 0
    with expected_warnings(['When min_distance < 1']):
        assert len(peak.peak_local_max(image, min_distance=0)) == image.size - 1