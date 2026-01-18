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
def update_solver_timelimit(opt, solver_name, timing, config):
    """Updates the time limit for subsolvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    timing : Timing
        Timing
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    elapsed = get_main_elapsed_time(timing)
    remaining = math.ceil(max(config.time_limit - elapsed, 1))
    if solver_name in {'cplex', 'appsi_cplex', 'cplex_persistent', 'gurobi', 'gurobi_persistent', 'appsi_gurobi'}:
        opt.options['timelimit'] = remaining
    elif solver_name == 'appsi_highs':
        opt.config.time_limit = remaining
    elif solver_name == 'cyipopt':
        opt.config.options['max_cpu_time'] = float(remaining)
    elif solver_name == 'glpk':
        opt.options['tmlim'] = remaining
    elif solver_name == 'baron':
        opt.options['MaxTime'] = remaining
    elif solver_name in {'ipopt', 'appsi_ipopt'}:
        opt.options['max_cpu_time'] = remaining
    elif solver_name == 'gams':
        opt.options['add_options'].append('option Reslim=%s;' % remaining)