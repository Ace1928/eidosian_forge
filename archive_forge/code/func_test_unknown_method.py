import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_unknown_method(self):
    method = 'foo'
    f0 = 10.0
    f1 = 20.0
    t1 = 1.0
    t = np.linspace(0, t1, 10)
    assert_raises(ValueError, waveforms.chirp, t, f0, t1, f1, method)