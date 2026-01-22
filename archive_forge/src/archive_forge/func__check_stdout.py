import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
def _check_stdout(self, output_string, exists=True):
    if exists:
        self.assertIn(self.characteristic_output_string, output_string)
    else:
        self.assertNotIn(self.characteristic_output_string, output_string)