import numpy as np
import pytest
from skimage.metrics import (
from skimage._shared.testing import (
def test_vi():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])
    assert_equal(np.sum(variation_of_information(im_true, im_test)), 1)