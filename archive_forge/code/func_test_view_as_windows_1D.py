import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_windows_1D():
    A = np.arange(10)
    window_shape = (3,)
    B = view_as_windows(A, window_shape)
    assert_equal(B, np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]))