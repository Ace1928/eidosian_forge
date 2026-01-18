import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_rectangle_perimiter():
    expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    start = (2, 3)
    end = (2, 4)
    img = np.zeros(expected.shape, dtype=np.uint8)
    rr, cc = rectangle_perimeter(start, end, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros(expected.shape, dtype=np.uint8)
    rr, cc = rectangle_perimeter(end=start, start=end, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros(expected.shape, dtype=np.uint8)
    start = (2, 3)
    extent = (1, 2)
    rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)