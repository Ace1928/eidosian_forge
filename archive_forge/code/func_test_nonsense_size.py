import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
@pytest.mark.parametrize('size', [0, -5, 3.5, 4.0])
def test_nonsense_size(size):
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(size, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(size, 0.2, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(3, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(3, 0.2, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt([1, 2, 3], size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        zoom_fft([1, 2, 3], 0.2, size)