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
class TestAutoscaleIP(AutoscaleTests):
    method = 'interior-point'

    def test_bug_6139(self):
        self.options['tol'] = 1e-10
        return AutoscaleTests.test_bug_6139(self)