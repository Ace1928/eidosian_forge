import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_watershed_ift09(self):
    data = np.array([[np.iinfo(np.uint16).max, 0], [0, 0]], np.uint16)
    markers = np.array([[1, 0], [0, 0]], np.int8)
    out = ndimage.watershed_ift(data, markers)
    expected = [[1, 1], [1, 1]]
    assert_allclose(out, expected)