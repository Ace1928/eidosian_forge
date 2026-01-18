import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_param_changes_4(self):
    with SolverFactory('gurobi_direct', manage_env=True) as opt:
        with self.assertRaisesRegex(gp.GurobiError, 'Unable to set'):
            opt.solve(self.model, options={'Method': -100})