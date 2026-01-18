import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_sepfir2d_invalid_filter():
    filt = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    image = np.random.rand(7, 9)
    signal.sepfir2d(image, filt, filt[2:])
    with pytest.raises(ValueError, match='odd length'):
        signal.sepfir2d(image, filt, filt[1:])
    with pytest.raises(ValueError, match='odd length'):
        signal.sepfir2d(image, filt[1:], filt)
    with pytest.raises(ValueError, match='object too deep'):
        signal.sepfir2d(image, filt.reshape(1, -1), filt)
    with pytest.raises(ValueError, match='object too deep'):
        signal.sepfir2d(image, filt, filt.reshape(1, -1))