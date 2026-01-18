import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
@pytest.mark.parametrize('shape', [(8, 16), (16, 8), (2, 3, 4)])
def test_window_shape_anisotropic(shape):
    w = window('hann', shape)
    assert w.shape == shape