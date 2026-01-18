import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_too_few_dimensions():
    bad = np.random.rand(4, 4).ravel()
    cb = np.random.rand(4)
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=bad, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=bad, b_eq=cb))