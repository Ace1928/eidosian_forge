import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
@pytest.mark.parametrize('sys,sample_time,samples_number', cases)
def test_step_invariant(self, sys, sample_time, samples_number):
    time = np.arange(samples_number) * sample_time
    _, yout_cont = step(sys, T=time)
    _, yout_disc = dstep(c2d(sys, sample_time, method='zoh'), n=len(time))
    assert_allclose(yout_cont.ravel(), yout_disc[0].ravel())