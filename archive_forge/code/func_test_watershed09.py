import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_watershed09(self):
    """Test on an image of reasonable size

        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
    image = np.zeros((1000, 1000))
    coords = np.random.uniform(0, 1000, (100, 2)).astype(int)
    markers = np.zeros((1000, 1000), int)
    idx = 1
    for x, y in coords:
        image[x, y] = 1
        markers[x, y] = idx
        idx += 1
    image = gaussian(image, sigma=4, mode='reflect')
    watershed(image, markers, self.eight)
    ndi.watershed_ift(image.astype(np.uint16), markers, self.eight)