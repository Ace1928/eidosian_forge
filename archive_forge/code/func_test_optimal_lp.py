import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
@unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
def test_optimal_lp(self):
    self.model.X = Var(within=NonNegativeReals)
    self.model.Obj = Objective(expr=self.model.X, sense=minimize)
    results = self.opt.solve(self.model)
    self.assertEqual(0.0, results.problem.lower_bound)
    self.assertEqual(0.0, results.problem.upper_bound)
    self.assertEqual(ProblemSense.minimize, results.problem.sense)
    self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
    self.assertEqual('Model was solved to optimality (subject to tolerances), and an optimal solution is available.', results.solver.termination_message)
    self.assertEqual(SolverStatus.ok, results.solver.status)