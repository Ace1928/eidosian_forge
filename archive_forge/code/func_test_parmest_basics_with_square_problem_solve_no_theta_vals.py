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
def test_parmest_basics_with_square_problem_solve_no_theta_vals(self):
    for model_type, parmest_input in self.input.items():
        pest = parmest.Estimator(parmest_input['model'], self.data, parmest_input['theta_names'], self.objective_function)
        obj_at_theta = pest.objective_at_theta(initialize_parmest_model=True)
        objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
        self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
        self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
        self.assertAlmostEqual(cov.iloc[1, 1], 0.04193591, places=2)