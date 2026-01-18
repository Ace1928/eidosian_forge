import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
def test_prewitt_v_horizontal():
    """Vertical prewitt on a horizontal edge should be zero."""
    i, j = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt_v(image)
    assert_allclose(result, 0)