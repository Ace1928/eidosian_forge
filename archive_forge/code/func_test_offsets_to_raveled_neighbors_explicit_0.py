import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.morphology import _util
def test_offsets_to_raveled_neighbors_explicit_0():
    """Check reviewed example."""
    image_shape = (100, 200, 3)
    footprint = np.ones((3, 3, 3), dtype=bool)
    center = (1, 1, 1)
    offsets = _util._offsets_to_raveled_neighbors(image_shape, footprint, center)
    desired = np.array([-600, -3, -1, 1, 3, 600, -603, -601, -599, -597, -4, -2, 2, 4, 597, 599, 601, 603, -604, -602, -598, -596, 596, 598, 602, 604])
    assert_array_equal(offsets, desired)