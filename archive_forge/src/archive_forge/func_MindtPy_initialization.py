import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def MindtPy_initialization(self):
    """Initializes the decomposition algorithm.

        This function initializes the decomposition algorithm, which includes generating the
        initial cuts required to build the main MIP.
        """
    config = self.config
    if config.init_strategy == 'rNLP':
        self.init_rNLP()
    elif config.init_strategy == 'max_binary':
        self.init_max_binaries()
    elif config.init_strategy == 'initial_binary':
        try:
            self.curr_int_sol = get_integer_solution(self.working_model)
        except TypeError as e:
            config.logger.error(e, exc_info=True)
            raise ValueError('The initial integer combination is not provided or not complete. Please provide the complete integer combination or use other initialization strategy.')
        self.integer_list.append(self.curr_int_sol)
        fixed_nlp, fixed_nlp_result = self.solve_subproblem()
        self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)
        self.integer_solution_to_cuts_index[self.curr_int_sol] = [1, len(self.mip.MindtPy_utils.cuts.oa_cuts)]
    elif config.init_strategy == 'FP':
        self.init_rNLP()
        self.fp_loop()