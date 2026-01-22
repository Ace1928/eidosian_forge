import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math
def add_var_bound(model, config):
    """This function will add bounds for variables in nonlinear constraints if they are not bounded.

    This is to avoid an unbounded main problem in the LP/NLP algorithm. Thus, the model will be
    updated to include bounds for the unbounded variables in nonlinear constraints.

    Parameters
    ----------
    model : PyomoModel
        Target model to add bound for its variables.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    MindtPy = model.MindtPy_utils
    for c in MindtPy.nonlinear_constraint_list:
        for var in EXPR.identify_variables(c.body):
            if var.has_lb() and var.has_ub():
                continue
            if not var.has_lb():
                if var.is_integer():
                    var.setlb(-config.integer_var_bound - 1)
                else:
                    var.setlb(-config.continuous_var_bound - 1)
            if not var.has_ub():
                if var.is_integer():
                    var.setub(config.integer_var_bound)
                else:
                    var.setub(config.continuous_var_bound)