import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_prewitt_mask(dtype):
    """Prewitt on a masked array should be zero."""
    result = filters.prewitt(np.random.uniform(size=(10, 10)).astype(dtype), np.zeros((10, 10), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert_allclose(np.abs(result), 0)