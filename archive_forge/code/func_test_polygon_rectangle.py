import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_polygon_rectangle():
    img = np.zeros((10, 10), 'uint8')
    poly = np.array(((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))
    rr, cc = polygon(poly[:, 0], poly[:, 1])
    img[rr, cc] = 1
    img_ = np.zeros((10, 10), 'uint8')
    img_[1:5, 1:5] = 1
    assert_array_equal(img, img_)