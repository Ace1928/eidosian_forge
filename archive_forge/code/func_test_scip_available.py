import subprocess
import sys
from os.path import join, exists, splitext
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
import pyomo.solvers.plugins.solvers.SCIPAMPL
def test_scip_available(self):
    self.set_solvers()
    scip = SolverFactory('scip', solver_io='nl')
    scip_executable = scip.executable()
    self.assertIs(scip_executable, self.executable_paths['scip'])
    self.assertEqual(1, self.path.call_count)
    self.assertEqual(1, self.run.call_count)
    self.available.assert_called()
    scip.executable()
    self.assertEqual(1, self.run.call_count)
    self.assertTrue(scip.available())