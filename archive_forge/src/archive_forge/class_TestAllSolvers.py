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
class TestAllSolvers(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_installed_solvers(self) -> None:
        """Test the list of installed solvers.
        """
        from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, SOLVER_MAP_CONIC, SOLVER_MAP_QP
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
        for solver in SOLVER_MAP_CONIC.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertAlmostEqual(prob.value, 1.0)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % solver)
        for solver in SOLVER_MAP_QP.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % solver)

    def test_mixed_integer_behavior(self) -> None:
        x = cp.Variable(2, name='x', integer=True)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [x >= 0])
        if INSTALLED_MI_SOLVERS == [cp.ECOS_BB]:
            with pytest.raises(cp.error.SolverError, match='You need a mixed-integer solver for this model'):
                prob.solve()
        else:
            prob.solve()
            self.assertItemsAlmostEqual(x.value, [0, 0])