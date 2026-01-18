import numpy as np
from skimage._shared.testing import assert_array_equal, assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats, rectangle
from skimage._shared import testing
def test_rect_3d_extent():
    expected = np.array([[[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=np.uint8)
    img = np.zeros((4, 5, 5), dtype=np.uint8)
    start = (0, 0, 2)
    extent = (5, 2, 3)
    pp, rr, cc = rectangle(start, extent=extent, shape=img.shape)
    img[pp, rr, cc] = 1
    assert_array_equal(img, expected)