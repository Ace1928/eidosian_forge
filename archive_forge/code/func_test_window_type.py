import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
@pytest.mark.parametrize('wintype', [16, 'triang', ('tukey', 0.8)])
def test_window_type(wintype):
    w = window(wintype, (9, 9))
    assert w.ndim == 2
    assert w.shape[1:] == w.shape[:-1]
    assert np.allclose(w.sum(axis=0), w.sum(axis=1))