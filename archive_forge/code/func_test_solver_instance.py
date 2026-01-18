import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_solver_instance(self):
    """
        Testing that we get a specific solver instance
        """
    ans = SolverFactory('none')
    self.assertTrue(isinstance(ans, UnknownSolver))
    ans = SolverFactory('_mock_cbc')
    self.assertEqual(type(ans), MockCBC)
    ans = SolverFactory('_mock_cbc', name='mymock')
    self.assertEqual(type(ans), MockCBC)
    self.assertEqual(ans.name, 'mymock')