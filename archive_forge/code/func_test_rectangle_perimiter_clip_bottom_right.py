import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_rectangle_perimiter_clip_bottom_right():
    expected = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.uint8)
    img = np.zeros((5, 5), dtype=np.uint8)
    start = (2, 2)
    extent = (10, 10)
    rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape, clip=False)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    expected = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 1, 1, 1]], dtype=np.uint8)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle_perimeter(start, extent=extent, shape=img.shape, clip=True)
    img[rr, cc] = 1
    assert_array_equal(img, expected)