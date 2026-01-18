import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest
@pytest.mark.parametrize('offset', (0, -3))
@pytest.mark.parametrize('data_already_sequential', (False, True))
def test_relabel_sequential_nonpositive_offset(data_already_sequential, offset):
    if data_already_sequential:
        ar = np.array([1, 3, 0, 2, 5, 4])
    else:
        ar = np.array([1, 1, 5, 5, 8, 99, 42, 0])
    with pytest.raises(ValueError):
        relabel_sequential(ar, offset=offset)