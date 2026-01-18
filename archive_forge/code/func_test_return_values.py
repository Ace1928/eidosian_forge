from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def test_return_values(self):
    objval, thetavals, data_rec = self.pest.theta_est(return_values=['ca', 'cb', 'cc', 'cd', 'caf'])
    self.assertAlmostEqual(data_rec['cc'].loc[18], 893.84924, places=3)