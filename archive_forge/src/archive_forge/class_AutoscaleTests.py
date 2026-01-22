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
class AutoscaleTests:
    options = {'autoscale': True}
    test_bug_6139 = LinprogCommonTests.test_bug_6139
    test_bug_6690 = LinprogCommonTests.test_bug_6690
    test_bug_7237 = LinprogCommonTests.test_bug_7237