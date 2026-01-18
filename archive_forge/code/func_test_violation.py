import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_violation():

    def cons_f(x):
        return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])
    nlc = NonlinearConstraint(cons_f, [-1, -0.85], [2, 2])
    pc = PreparedConstraint(nlc, [0.5, 1])
    assert_array_equal(pc.violation([0.5, 1]), [0.0, 0.0])
    np.testing.assert_almost_equal(pc.violation([0.5, 1.2]), [0.0, 0.1])
    np.testing.assert_almost_equal(pc.violation([1.2, 1.2]), [0.64, 0])
    np.testing.assert_almost_equal(pc.violation([0.1, -1.2]), [0.19, 0])
    np.testing.assert_almost_equal(pc.violation([0.1, 2]), [0.01, 1.14])