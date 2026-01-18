import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
def test_version_gms(self):
    with SolverFactory('gams', solver_io='gms') as opt:
        self.assertIsNotNone(opt.version())