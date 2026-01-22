import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
class TestLinprogSimplexBland(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'bland': True}

    def test_bug_5400(self):
        pytest.skip('Simplex fails on this problem.')

    def test_bug_8174_low_tol(self):
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError):
            with pytest.warns(OptimizeWarning):
                super().test_bug_8174()