import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_unsupported_int64(self):
    for dtype in (np.int64, np.uint64):
        image = np.zeros((10, 10), dtype=dtype)
        image[3, 3] = np.iinfo(dtype).max
        with pytest.raises(ValueError, match='64-bit integer images are not supported'):
            feature.canny(image)