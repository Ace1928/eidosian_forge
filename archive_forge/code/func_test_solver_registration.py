import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_solver_registration(self):
    """
        Testing methods in the solverwriter factory registration process
        """
    SolverFactory.unregister('stest3')
    self.assertTrue('stest3' not in SolverFactory)
    SolverFactory.register('stest3')(MockSolver)
    self.assertTrue('stest3' in SolverFactory)
    self.assertTrue('_mock_cbc' in SolverFactory)