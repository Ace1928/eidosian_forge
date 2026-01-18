import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_sepfir2d_invalid_image():
    filt = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    image = np.random.rand(8, 8)
    with pytest.raises(ValueError, match='object too deep'):
        signal.sepfir2d(image.reshape(4, 4, 4), filt, filt)
    with pytest.raises(ValueError, match='object of too small depth'):
        signal.sepfir2d(image[0], filt, filt)