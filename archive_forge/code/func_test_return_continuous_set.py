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
def test_return_continuous_set(self):
    """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
    obj1, theta1, return_vals1 = self.pest_df.theta_est(return_values=['time'])
    obj2, theta2, return_vals2 = self.pest_dict.theta_est(return_values=['time'])
    self.assertAlmostEqual(return_vals1['time'].loc[0][18], 2.368, places=3)
    self.assertAlmostEqual(return_vals2['time'].loc[0][18], 2.368, places=3)