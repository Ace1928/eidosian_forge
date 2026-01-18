from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_nonequal_timesteps(self):
    t = np.array([0.0, 1.0, 1.0, 3.0])
    u = np.array([0.0, 0.0, 1.0, 1.0])
    system = ([1.0], [1.0, 0.0])
    with assert_raises(ValueError, match='Time steps are not equally spaced.'):
        tout, y, x = self.func(system, u, t, X0=[1.0])