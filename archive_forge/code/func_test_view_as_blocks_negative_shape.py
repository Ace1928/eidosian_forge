import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_blocks_negative_shape():
    A = np.arange(10)
    with testing.raises(ValueError):
        view_as_blocks(A, (-2,))