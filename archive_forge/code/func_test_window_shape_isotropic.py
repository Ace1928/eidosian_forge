import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
@pytest.mark.parametrize('size', [5, 6])
@pytest.mark.parametrize('ndim', [2, 3, 4])
def test_window_shape_isotropic(size, ndim):
    w = window('hann', (size,) * ndim)
    assert w.ndim == ndim
    assert w.shape[1:] == w.shape[:-1]
    for i in range(1, ndim - 1):
        assert np.allclose(w.sum(axis=0), w.sum(axis=i))