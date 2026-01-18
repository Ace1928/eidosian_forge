import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_two_objects(self):
    image = np.zeros((10, 20))
    labels = np.zeros((10, 20), int)
    image[5, 5] = 1
    image[5, 15] = 0.5
    labels[5, 5] = 1
    labels[5, 15] = 2
    expected = np.stack(np.where(labels > 0), axis=-1)
    result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
    assert_array_equal(result, expected)