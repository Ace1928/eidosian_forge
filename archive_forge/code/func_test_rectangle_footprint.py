import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_rectangle_footprint(self):
    """Test rectangle footprints"""
    for i in range(0, 5):
        for j in range(0, 5):
            actual_mask = footprints.rectangle(i, j)
            expected_mask = np.ones((i, j), dtype='uint8')
            assert_equal(expected_mask, actual_mask)