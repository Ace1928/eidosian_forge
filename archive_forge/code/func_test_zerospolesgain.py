import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_zerospolesgain(self):
    zeros_c = np.array([0.5, -0.5])
    poles_c = np.array([1j / np.sqrt(2), -1j / np.sqrt(2)])
    k_c = 1.0
    zeros_d = [1.2337172730586, 0.735356894461267]
    polls_d = [0.938148335039729 + 0.346233593780536j, 0.938148335039729 - 0.346233593780536j]
    k_d = 1.0
    dt_requested = 0.5
    zeros, poles, k, dt = c2d((zeros_c, poles_c, k_c), dt_requested, method='zoh')
    assert_array_almost_equal(zeros_d, zeros)
    assert_array_almost_equal(polls_d, poles)
    assert_almost_equal(k_d, k)
    assert_almost_equal(dt_requested, dt)