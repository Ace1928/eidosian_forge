import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_watershed11(self):
    """Make sure that all points on this plateau are assigned to closest seed"""
    image = np.zeros((21, 21))
    markers = np.zeros((21, 21), int)
    markers[5, 5] = 1
    markers[5, 10] = 2
    markers[10, 5] = 3
    markers[10, 10] = 4
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
    out = watershed(image, markers, structure)
    i, j = np.mgrid[0:21, 0:21]
    d = np.dstack([np.sqrt((i.astype(float) - i0) ** 2, (j.astype(float) - j0) ** 2) for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))])
    dmin = np.min(d, 2)
    self.assertTrue(np.all(d[i, j, out[i, j] - 1] == dmin))