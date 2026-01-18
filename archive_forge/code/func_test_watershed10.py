import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_watershed10(self):
    """watershed 10"""
    data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], np.uint8)
    markers = np.array([[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]], np.int8)
    out = watershed(data, markers, self.eight)
    error = diff([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], out)
    self.assertTrue(error < eps)