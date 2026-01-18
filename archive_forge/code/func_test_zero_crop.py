import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_zero_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, 0)
    assert out.shape == (9, 5)