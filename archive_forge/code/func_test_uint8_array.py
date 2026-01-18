import numpy as np
from skimage.util import unique_rows
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_uint8_array():
    ar = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0, 1, 0], [1, 0, 1]], np.uint8)
    assert_equal(ar_out, desired_ar_out)