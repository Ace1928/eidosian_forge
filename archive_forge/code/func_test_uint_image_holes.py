import numpy as np
import pytest
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings
def test_uint_image_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]], dtype=np.uint8)
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=bool)
    with expected_warnings(['returned as a boolean array']):
        observed = remove_small_holes(labeled_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)