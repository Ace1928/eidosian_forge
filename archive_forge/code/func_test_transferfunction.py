import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_transferfunction(self):
    numc = np.array([0.25, 0.25, 0.5])
    denc = np.array([0.75, 0.75, 1.0])
    numd = np.array([[1.0 / 3.0, -0.427419169438754, 0.221654141101125]])
    dend = np.array([1.0, -1.351394049721225, 0.606530659712634])
    dt_requested = 0.5
    num, den, dt = c2d((numc, denc), dt_requested, method='zoh')
    assert_array_almost_equal(numd, num)
    assert_array_almost_equal(dend, den)
    assert_almost_equal(dt_requested, dt)