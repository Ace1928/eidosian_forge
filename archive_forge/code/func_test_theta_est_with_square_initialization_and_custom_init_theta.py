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
def test_theta_est_with_square_initialization_and_custom_init_theta(self):
    theta_vals_init = pd.DataFrame(data=[[19.0, 0.5]], columns=['asymptote', 'rate_constant'])
    obj_init = self.pest.objective_at_theta(theta_values=theta_vals_init, initialize_parmest_model=True)
    objval, thetavals = self.pest.theta_est()
    self.assertAlmostEqual(objval, 4.3317112, places=2)
    self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2)
    self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2)