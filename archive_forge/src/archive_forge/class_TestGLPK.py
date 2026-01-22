import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
@unittest.skipUnless('GLPK' in INSTALLED_SOLVERS, 'GLPK is not installed.')
class TestGLPK(unittest.TestCase):

    def test_glpk_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GLPK')

    def test_glpk_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GLPK')

    def test_glpk_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GLPK')

    def test_glpk_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='GLPK')

    def test_glpk_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GLPK')

    def test_glpk_lk_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GLPK')

    def test_glpk_lp_6(self) -> None:
        StandardTestLPs.test_lp_6(solver='GLPK')

    def test_glpk_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='GLPK_MI')

    def test_glpk_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='GLPK_MI')

    def test_glpk_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='GLPK_MI')

    def test_glpk_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='GLPK_MI')

    def test_glpk_mi_lp_4(self) -> None:
        StandardTestLPs.test_mi_lp_4(solver='GLPK_MI')

    def test_glpk_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='GLPK_MI')

    def test_glpk_options(self) -> None:
        sth = sths.lp_1()
        import cvxopt
        assert 'tm_lim' not in cvxopt.glpk.options
        sth.solve(solver='GLPK', tm_lim=100)
        assert 'tm_lim' not in cvxopt.glpk.options
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)

    def test_glpk_mi_options(self) -> None:
        sth = sths.mi_lp_1()
        import cvxopt
        assert 'tm_lim' not in cvxopt.glpk.options
        sth.solve(solver='GLPK_MI', tm_lim=100, verbose=True)
        assert 'tm_lim' not in cvxopt.glpk.options
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)