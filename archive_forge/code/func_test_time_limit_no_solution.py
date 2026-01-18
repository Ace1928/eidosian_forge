import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
@gurobi_test
def test_time_limit_no_solution(self):
    print('\t Test time limit with no solution')
    time_limit = 1
    solver_settings = dict(HiGHS=50, PULP_CBC_CMD=30, COIN_CMD=30)
    bins = solver_settings.get(self.solver.name)
    if bins is None:
        return
    prob = create_bin_packing_problem(bins=bins, seed=99)
    self.solver.timeLimit = time_limit
    status = prob.solve(self.solver)
    self.assertEqual(prob.status, const.LpStatusNotSolved)
    self.assertEqual(status, const.LpStatusNotSolved)
    self.assertEqual(prob.sol_status, const.LpSolutionNoSolutionFound)