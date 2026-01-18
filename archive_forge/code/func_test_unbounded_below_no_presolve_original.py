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
def test_unbounded_below_no_presolve_original(self):
    c = [-1]
    bounds = [(None, 1)]
    res = linprog(c=c, bounds=bounds, method=self.method, options={'presolve': False, 'cholesky': True})
    _assert_success(res, desired_fun=-1)