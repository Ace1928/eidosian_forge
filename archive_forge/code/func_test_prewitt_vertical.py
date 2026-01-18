import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_prewitt_vertical():
    """Prewitt on a vertical edge should be a vertical line."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert_allclose(result[np.abs(j) > 1], 0, atol=1e-10)