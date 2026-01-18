import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_np_int_crop():
    arr = np.arange(45).reshape(9, 5)
    out1 = crop(arr, np.int64(1))
    out2 = crop(arr, np.int32(1))
    assert_array_equal(out1, out2)
    assert out1.shape == (7, 3)