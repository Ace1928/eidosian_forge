import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_watershed_input_not_modified(self):
    """Test to ensure input markers are not modified."""
    image = np.random.default_rng().random(size=(21, 21))
    markers = np.zeros((21, 21), dtype=np.uint8)
    markers[[5, 5, 15, 15], [5, 15, 5, 15]] = [1, 2, 3, 4]
    original_markers = np.copy(markers)
    result = watershed(image, markers)
    np.testing.assert_equal(original_markers, markers)
    assert not np.all(result == markers)