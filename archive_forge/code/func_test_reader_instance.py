import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_reader_instance(self):
    """
        Testing that we get a specific reader instance
        """
    ans = ReaderFactory('none')
    self.assertEqual(ans, None)
    ans = ReaderFactory('sol')
    self.assertEqual(type(ans), ResultsReader_sol)