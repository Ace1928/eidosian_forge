import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_sweep_poly_cubic(self):
    p = np.poly1d([2.0, 1.0, 0.0, -2.0])
    t = np.linspace(0, 2.0, 10000)
    phase = waveforms._sweep_poly_phase(t, p)
    tf, f = compute_frequency(t, phase)
    expected = p(tf)
    abserr = np.max(np.abs(f - expected))
    assert_(abserr < 1e-06)