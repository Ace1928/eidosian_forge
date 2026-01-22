from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def _assert_poles_close(P1, P2, rtol=1e-08, atol=1e-08):
    """
    Check each pole in P1 is close to a pole in P2 with a 1e-8
    relative tolerance or 1e-8 absolute tolerance (useful for zero poles).
    These tolerances are very strict but the systems tested are known to
    accept these poles so we should not be far from what is requested.
    """
    P2 = P2.copy()
    for p1 in P1:
        found = False
        for p2_idx in range(P2.shape[0]):
            if np.allclose([np.real(p1), np.imag(p1)], [np.real(P2[p2_idx]), np.imag(P2[p2_idx])], rtol, atol):
                found = True
                np.delete(P2, p2_idx)
                break
        if not found:
            raise ValueError("Can't find pole " + str(p1) + ' in ' + str(P2))