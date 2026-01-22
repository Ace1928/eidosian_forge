import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
class TestKKT_QPs(BaseTest):

    def test_qp_0(self, places=4):
        sth = STH.qp_0()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth