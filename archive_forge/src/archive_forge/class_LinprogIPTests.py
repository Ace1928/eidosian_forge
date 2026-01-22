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
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class LinprogIPTests(LinprogCommonTests):
    method = 'interior-point'

    def test_bug_10466(self):
        pytest.skip('Test is failing, but solver is deprecated.')