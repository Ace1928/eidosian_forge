import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_quad_psd(self) -> None:
    """Test PSD checking from #1491.
        """
    x = cp.Variable(2, complex=True)
    P1 = np.eye(2)
    P2 = np.array([[1 + 0j, 0 + 0j], [0 - 0j, 1 + 0j]])
    print('P1 is real:', cp.quad_form(x, P1).curvature)
    print('P2 is complex:', cp.quad_form(x, P2).curvature)
    assert cp.quad_form(x, P2).is_dcp()