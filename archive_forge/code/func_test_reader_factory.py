import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_reader_factory(self):
    """
        Testing the pyomo.opt reader factory
        """
    ReaderFactory.register('rtest3')(MockReader)
    ans = ReaderFactory
    self.assertTrue(set(ans) >= set(['rtest3', 'sol', 'yaml', 'json']))