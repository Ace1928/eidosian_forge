import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
def test_process_logfile(self):
    cbc_shell = CBCSHELL()
    cbc_shell._log_file = os.path.join(data_dir, 'test5_timeout_cbc.txt')
    results = cbc_shell.process_logfile()
    self.assertEqual(results.solution.gap, 0.01)
    self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 50364)
    self.assertEqual(results.solver.system_time, 2.01)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 34776)