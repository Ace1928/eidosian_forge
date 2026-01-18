import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_pair_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2),))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert_equal(out.shape, (6, 2))