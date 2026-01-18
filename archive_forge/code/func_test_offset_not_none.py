import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_offset_not_none():
    """Test reconstruction with valid offset parameter"""
    seed = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = np.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    expected = np.array([0, 3, 6, 6, 6, 6, 6, 4, 4, 0])
    assert_array_almost_equal(reconstruction(seed, mask, method='dilation', footprint=np.ones(3), offset=np.array([0])), expected)