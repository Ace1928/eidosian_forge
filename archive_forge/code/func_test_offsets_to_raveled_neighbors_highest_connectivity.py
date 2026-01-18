import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.morphology import _util
@pytest.mark.parametrize('image_shape', [(111,), (33, 44), (22, 55, 11), (6, 5, 4, 3)])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_offsets_to_raveled_neighbors_highest_connectivity(image_shape, order):
    """
    Check scenarios where footprint is always of the highest connectivity
    and all dimensions are > 2.
    """
    footprint = np.ones((3,) * len(image_shape), dtype=bool)
    center = (1,) * len(image_shape)
    offsets = _util._offsets_to_raveled_neighbors(image_shape, footprint, center, order)
    assert len(offsets) == footprint.sum() - 1
    assert 0 not in offsets
    assert len(set(offsets)) == offsets.size
    assert all((-x in offsets for x in offsets))
    image_center = tuple((s // 2 for s in image_shape))
    coords = [np.abs(np.arange(s, dtype=np.intp) - c) for s, c in zip(image_shape, image_center)]
    grid = np.meshgrid(*coords, indexing='ij')
    image = np.sum(grid, axis=0)
    image_raveled = image.ravel(order)
    image_center_raveled = np.ravel_multi_index(image_center, image_shape, order=order)
    samples = []
    for offset in offsets:
        index = image_center_raveled + offset
        samples.append(image_raveled[index])
    assert np.min(samples) == 1
    assert np.max(samples) == len(image_shape)
    assert list(sorted(samples)) == samples