import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_writer_instance(self):
    """
        Testing that we get a specific writer instance

        Note: this simply provides code coverage right now, but
        later it should be adapted to generate a specific writer.
        """
    ans = WriterFactory('none')
    self.assertEqual(ans, None)
    ans = WriterFactory('wtest3')
    self.assertNotEqual(ans, None)