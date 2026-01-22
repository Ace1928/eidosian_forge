import logging
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_FP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.core import ConstraintList
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts
Deactivate the nonlinear constraints to create the MIP problem.