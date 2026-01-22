import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
class TestKKT_SDPs(BaseTest):

    def test_sdp_1min(self, places=4):
        sth = STH.sdp_1('min')
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_sdp_1max(self, places=4):
        sth = STH.sdp_1('max')
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_sdp_2(self, places=4):
        sth = STH.sdp_2()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth