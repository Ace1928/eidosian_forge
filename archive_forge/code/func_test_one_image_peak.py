import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = np.ones((5, 5))
    image[2, 2] = 2
    mask = np.ones((5, 5)) * 3
    assert_array_almost_equal(reconstruction(image, mask), 2)