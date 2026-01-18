import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
def test_pcp_4(self, places: int=3):
    sth = self.non_vec_pow_nd()
    sth.solve(solver='SCS', eps=1e-06)
    sth.check_primal_feasibility(places)
    sth.check_complementarity(places)
    sth.check_dual_domains(places)
    sth.check_stationary_lagrangian(places)
    return sth