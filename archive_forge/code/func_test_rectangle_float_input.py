import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_rectangle_float_input():
    expected = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    start = (0.2, 0.8)
    end = (3.1, 2.9)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle(start, end=end, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle(end=start, start=end, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle(start=(3.1, 0.8), end=(0.2, 2.9), shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle(end=(3.1, 0.8), start=(0.2, 2.9), shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)