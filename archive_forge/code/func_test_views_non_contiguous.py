import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util.shape import view_as_blocks, view_as_windows
def test_views_non_contiguous():
    A = np.arange(16).reshape((4, 4))
    A = A[::2, :]
    res_b = view_as_blocks(A, (2, 2))
    res_w = view_as_windows(A, (2, 2))
    print(res_b)
    print(res_w)
    expected_b = [[[[0, 1], [8, 9]], [[2, 3], [10, 11]]]]
    expected_w = [[[[0, 1], [8, 9]], [[1, 2], [9, 10]], [[2, 3], [10, 11]]]]
    assert_equal(res_b, expected_b)
    assert_equal(res_w, expected_w)