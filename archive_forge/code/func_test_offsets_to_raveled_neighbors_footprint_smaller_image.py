import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.morphology import _util
@pytest.mark.parametrize('image_shape', [(2,), (2, 2), (2, 1, 2), (2, 2, 1, 2), (0, 2, 1, 2)])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_offsets_to_raveled_neighbors_footprint_smaller_image(image_shape, order):
    """
    Test if a dimension indicated by `image_shape` is smaller than in
    `footprint`.
    """
    footprint = np.ones((3,) * len(image_shape), dtype=bool)
    center = (1,) * len(image_shape)
    offsets = _util._offsets_to_raveled_neighbors(image_shape, footprint, center, order)
    assert len(offsets) <= footprint.sum() - 1
    assert 0 not in offsets
    assert len(set(offsets)) == offsets.size
    assert all((-x in offsets for x in offsets))