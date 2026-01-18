import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_peaks_in_contact(self):
    image = np.zeros((15, 15))
    x0, y0, i0 = (8, 8, 1)
    x1, y1, i1 = (7, 7, 1)
    x2, y2, i2 = (6, 6, 1)
    image[y0, x0] = i0
    image[y1, x1] = i1
    image[y2, x2] = i2
    out = peak._prominent_peaks(image, min_xdistance=3, min_ydistance=3)
    assert_equal(out[0], np.array((i1,)))
    assert_equal(out[1], np.array((x1,)))
    assert_equal(out[2], np.array((y1,)))