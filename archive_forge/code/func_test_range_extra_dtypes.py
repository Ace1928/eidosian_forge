import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
@parametrize('dtype_in, dt', dtype_pairs)
def test_range_extra_dtypes(dtype_in, dt):
    """Test code paths that are not skipped by `test_range`"""
    imin, imax = dtype_range_extra[dtype_in]
    x = np.linspace(imin, imax, 10).astype(dtype_in)
    y = _convert(x, dt)
    omin, omax = dtype_range_extra[dt]
    _verify_range(f'From {np.dtype(dtype_in)} to {np.dtype(dt)}', y, omin, omax, np.dtype(dt))