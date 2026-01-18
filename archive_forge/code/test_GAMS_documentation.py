import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
Check for logfiles existence and contents.

        exists=True:
            Whether to check if the logfile exists or doesn't exist.
        expected=None:
            Optionally check that the logfiles contents is equal to this value.

        