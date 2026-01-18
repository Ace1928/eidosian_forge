import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
def test_numeric_seed_watershed():
    """Test that passing just the number of seeds to watershed works."""
    image = np.zeros((5, 6))
    image[:, 3:] = 1
    compact = watershed(image, 2, compactness=0.01)
    expected = np.array([[1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2]], dtype=np.int32)
    np.testing.assert_equal(compact, expected)