import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_reorder_labels(self):
    image = np.random.uniform(size=(40, 60))
    i, j = np.mgrid[0:40, 0:60]
    labels = 1 + (i >= 20) + (j >= 30) * 2
    labels[labels == 4] = 5
    i, j = np.mgrid[-3:4, -3:4]
    footprint = i * i + j * j <= 9
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 20), (20, 40)):
        for jmin, jmax in ((0, 30), (30, 60)):
            expected[imin:imax, jmin:jmax] = ndi.maximum_filter(image[imin:imax, jmin:jmax], footprint=footprint)
    expected = expected == image
    peak_idx = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, footprint=footprint, exclude_border=False)
    result = np.zeros_like(expected, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert (result == expected).all()