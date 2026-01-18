import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings
def test_pathological_qhull_example():
    image = np.array([[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]], dtype=bool)
    expected = np.array([[0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=bool)
    assert_array_equal(convex_hull_image(image), expected)