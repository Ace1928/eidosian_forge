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
@unittest.skipUnless('CLARABEL' in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
class TestClarabel(BaseTest):
    """ Unit tests for Clarabel. """

    def setUp(self) -> None:
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_clarabel_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver=cp.CLARABEL)

    def test_clarabel_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CLARABEL')

    def test_clarabel_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CLARABEL')

    def test_clarabel_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='CLARABEL')

    def test_clarabel_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CLARABEL')

    def test_clarabel_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CLARABEL')

    def test_clarabel_qp_0(self) -> None:
        StandardTestQPs.test_qp_0(solver='CLARABEL')

    def test_clarabel_qp_0_linear_obj(self) -> None:
        StandardTestQPs.test_qp_0(solver='CLARABEL', use_quad_obj=False)

    def test_clarabel_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CLARABEL')

    def test_clarabel_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CLARABEL')

    def test_clarabel_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CLARABEL')

    def test_clarabel_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='CLARABEL')
        StandardTestSOCPs.test_socp_3ax1(solver='CLARABEL')

    def test_clarabel_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='CLARABEL')

    def test_clarabel_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='CLARABEL')

    def test_clarabel_pcp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CLARABEL')

    def test_clarabel_pcp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CLARABEL')

    def test_clarabel_pcp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CLARABEL')

    def test_clarabel_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='CLARABEL')

    def test_clarabel_sdp_2(self) -> None:
        places = 3
        sth = sths.sdp_2()
        sth.solve('CLARABEL')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)