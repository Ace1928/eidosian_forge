import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask
@pytest.mark.parametrize('shape,channel_axis', [((16, 16), None), ((15, 15, 2), -1), ((13, 17, 3), -1)])
@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, preserve, dtype):
    radius = 2.0
    amount = 1.0
    array = (np.random.random(shape) * 10).astype(dtype, copy=False)
    negative = np.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve, channel_axis=channel_axis)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype == _supported_float_type(dtype)
    assert output.shape == shape