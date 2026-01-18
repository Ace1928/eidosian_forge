import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask
@pytest.mark.parametrize('shape,channel_axis', [((16, 16), None), ((15, 15, 2), -1), ((13, 17, 3), -1), ((2, 15, 15), 0), ((3, 13, 17), 0)])
@pytest.mark.parametrize('offset', [-5, 0, 5])
@pytest.mark.parametrize('preserve', [False, True])
def test_unsharp_masking_with_different_ranges(shape, offset, channel_axis, preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve, channel_axis=channel_axis)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape