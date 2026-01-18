import pytest
from skimage._shared._geometry import polygon_clip, polygon_area
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
def test_poly_clip():
    x = [0, 1, 2, 1]
    y = [0, -1, 0, 1]
    yc, xc = polygon_clip(y, x, 0, 0, 1, 1)
    assert_equal(polygon_area(yc, xc), 0.5)
    x = [-1, 1.5, 1.5, -1]
    y = [0.5, 0.5, 1.5, 1.5]
    yc, xc = polygon_clip(y, x, 0, 0, 1, 1)
    assert_equal(polygon_area(yc, xc), 0.5)