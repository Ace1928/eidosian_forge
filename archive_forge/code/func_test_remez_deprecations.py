import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_remez_deprecations(self):
    with pytest.deprecated_call(match="'remez' keyword argument 'Hz'"):
        remez(12, [0, 0.3, 0.5, 1], [1, 0], Hz=2.0)
    with pytest.deprecated_call(match='use keyword arguments'):
        remez(11, [0.1, 0.4], [1], None)