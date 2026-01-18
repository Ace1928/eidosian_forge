import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_view_as_windows_with_skip():
    A = np.arange(20).reshape((5, 4))
    B = view_as_windows(A, 2, step=2)
    assert_equal(B, [[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10, 11], [14, 15]]]])
    C = view_as_windows(A, 2, step=4)
    assert_equal(C.shape, (1, 1, 2, 2))