import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_signed_scaling_float32():
    x = np.array([-128, 127], dtype=np.int8)
    y = img_as_float32(x)
    assert_equal(y.max(), 1)