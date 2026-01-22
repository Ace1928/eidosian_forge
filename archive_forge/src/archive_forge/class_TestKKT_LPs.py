import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
class TestKKT_LPs(BaseTest):

    def test_lp_1(self, places=4):
        sth = STH.lp_1()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)

    def test_lp_2(self, places=4):
        sth = STH.lp_2()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)

    def test_lp_5(self, places=4):
        sth = STH.lp_5()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)