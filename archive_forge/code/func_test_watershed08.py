import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_watershed08(self):
    """The border pixels + an edge are all the same value"""
    data = blob.copy()
    data[10, 7:9] = 141
    mask = data != 255
    markers = np.zeros(data.shape, int)
    markers[6, 7] = 1
    markers[14, 7] = 2
    out = watershed(data, markers, self.eight, mask=mask)
    size1 = np.sum(out == 1)
    size2 = np.sum(out == 2)
    self.assertTrue(abs(size1 - size2) <= 6)