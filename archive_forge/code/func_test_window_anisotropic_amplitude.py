import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
@pytest.mark.parametrize('shape', [[17, 33], [17, 97]])
def test_window_anisotropic_amplitude(shape):
    w = window(('tukey', 0.8), shape)
    profile_w = w[w.shape[0] // 2, :]
    profile_h = w[:, w.shape[1] // 2]
    assert abs(profile_w.mean() - profile_h.mean()) < 0.01