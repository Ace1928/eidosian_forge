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
@unittest.skip("Most folks don't have k_aug installed")
def test_theta_k_aug_for_Hessian(self):
    objval, thetavals, Hessian = self.pest.theta_est(solver='k_aug')
    self.assertAlmostEqual(objval, 4.4675, places=2)