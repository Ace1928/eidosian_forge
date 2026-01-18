import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask
@pytest.mark.parametrize('shape,multichannel', [((29,), False), ((40, 4), True), ((32, 32), False), ((29, 31, 3), True), ((13, 17, 4, 8), False)])
@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('radius', [0, 0.1, 2.0])
@pytest.mark.parametrize('amount', [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize('offset', [-1.0, 0.0, 1.0])
@pytest.mark.parametrize('preserve', [False, True])
def test_unsharp_masking_output_type_and_shape(radius, amount, shape, multichannel, dtype, offset, preserve):
    array = np.random.random(shape)
    array = ((array + offset) * 128).astype(dtype)
    if preserve is False and dtype in [np.float32, np.float64]:
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve, channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape