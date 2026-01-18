import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.morphology import _util
def test_offsets_to_raveled_neighbors_explicit_1():
    """Check reviewed example where footprint is larger in last dimension."""
    image_shape = (10, 9, 8, 3)
    footprint = np.ones((3, 3, 3, 4), dtype=bool)
    center = (1, 1, 1, 1)
    offsets = _util._offsets_to_raveled_neighbors(image_shape, footprint, center)
    desired = np.array([-216, -24, -3, -1, 1, 3, 24, 216, -240, -219, -217, -215, -213, -192, -27, -25, -23, -21, -4, -2, 2, 4, 21, 23, 25, 27, 192, 213, 215, 217, 219, 240, -243, -241, -239, -237, -220, -218, -214, -212, -195, -193, -191, -189, -28, -26, -22, -20, 20, 22, 26, 28, 189, 191, 193, 195, 212, 214, 218, 220, 237, 239, 241, 243, -244, -242, -238, -236, -196, -194, -190, -188, 188, 190, 194, 196, 236, 238, 242, 244, 5, -211, -19, 29, 221, -235, -187, 197, 245])
    assert_array_equal(offsets, desired)