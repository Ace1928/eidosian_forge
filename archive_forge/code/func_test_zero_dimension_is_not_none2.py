from abc import abstractmethod
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from pytest import raises as assert_raises
from pytest import warns
from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
from scipy.signal._filter_design import BadCoefficients
import scipy.linalg as linalg
def test_zero_dimension_is_not_none2(self):
    B_ = np.zeros((2, 0))
    C_ = np.zeros((0, 2))
    A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
    assert_equal(A, self.A)
    assert_equal(B, B_)
    assert_equal(C, C_)
    assert_equal(D.shape[0], C_.shape[0])
    assert_equal(D.shape[1], B_.shape[1])