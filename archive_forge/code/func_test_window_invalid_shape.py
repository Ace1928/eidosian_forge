import numpy as np
import pytest
from scipy.signal import get_window
from skimage.filters import window
def test_window_invalid_shape():
    with pytest.raises(ValueError):
        window(10, shape=(-5, 10))
    with pytest.raises(ValueError):
        window(10, shape=(1.3, 2.0))