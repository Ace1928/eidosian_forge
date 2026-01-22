from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
class TestLti:

    def test_lti_instantiation(self):
        s = lti([1], [-1])
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)
        s = lti(np.array([]), np.array([-1]), 1)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)
        s = lti([], [-1], 1)
        s = lti([1], [-1], 1, 3)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)