import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_sobel_v_zeros():
    """Vertical sobel on an array of all zeros."""
    result = filters.sobel_v(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)