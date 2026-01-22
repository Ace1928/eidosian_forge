import json
import os
from os.path import dirname, abspath, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.kernel import SolverFactory, variable, maximize, minimize
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
class PiecewiseLinearKernelTest(unittest.TestCase):
    pass