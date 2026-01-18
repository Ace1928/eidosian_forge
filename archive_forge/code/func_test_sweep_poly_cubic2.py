import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_sweep_poly_cubic2(self):
    """Use an array of coefficients instead of a poly1d."""
    p = np.array([2.0, 1.0, 0.0, -2.0])
    t = np.linspace(0, 2.0, 10000)
    phase = waveforms._sweep_poly_phase(t, p)
    tf, f = compute_frequency(t, phase)
    expected = np.poly1d(p)(tf)
    abserr = np.max(np.abs(f - expected))
    assert_(abserr < 1e-06)