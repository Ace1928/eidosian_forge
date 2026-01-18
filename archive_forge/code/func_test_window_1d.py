import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
@pytest.mark.parametrize('size', [10, 11])
def test_window_1d(size):
    w = window('hann', size)
    w1 = get_window('hann', size, fftbins=False)
    assert np.allclose(w, w1)