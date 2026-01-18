import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_writer_registration(self):
    """
        Testing methods in the writer factory registration process
        """
    WriterFactory.unregister('wtest3')
    self.assertTrue(not 'wtest3' in WriterFactory)
    WriterFactory.register('wtest3')(MockWriter)
    self.assertTrue('wtest3' in WriterFactory)