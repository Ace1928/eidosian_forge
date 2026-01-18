import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_saturated_arithmetic(self):
    """Adding/subtracting a constant and clipping"""
    data = np.array([[250, 251, 5, 5], [100, 200, 253, 252], [4, 10, 1, 3]], dtype=np.uint8)
    img_constant_added = extrema._add_constant_clip(data, 4)
    expected = np.array([[254, 255, 9, 9], [104, 204, 255, 255], [8, 14, 5, 7]], dtype=np.uint8)
    error = diff(img_constant_added, expected)
    assert error < eps
    img_constant_subtracted = extrema._subtract_constant_clip(data, 4)
    expected = np.array([[246, 247, 1, 1], [96, 196, 249, 248], [0, 6, 0, 0]], dtype=np.uint8)
    error = diff(img_constant_subtracted, expected)
    assert error < eps
    data = np.array([[32767, 32766], [-32768, -32767]], dtype=np.int16)
    img_constant_added = extrema._add_constant_clip(data, 1)
    expected = np.array([[32767, 32767], [-32767, -32766]], dtype=np.int16)
    error = diff(img_constant_added, expected)
    assert error < eps
    img_constant_subtracted = extrema._subtract_constant_clip(data, 1)
    expected = np.array([[32766, 32765], [-32768, -32768]], dtype=np.int16)
    error = diff(img_constant_subtracted, expected)
    assert error < eps