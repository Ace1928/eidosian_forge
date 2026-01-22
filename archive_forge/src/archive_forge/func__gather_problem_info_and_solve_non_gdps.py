from io import StringIO
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigBlock
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.config_options import _add_common_configs
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.util import (
from pyomo.core.base import Objective, value, minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.util.model_size import build_model_size_report
def _gather_problem_info_and_solve_non_gdps(self, model, config):
    """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
    logger = config.logger
    self._create_pyomo_results_object_with_problem_info(model, config)
    problem = self.pyomo_results.problem
    if problem.number_of_binary_variables == 0 and problem.number_of_integer_variables == 0 and (problem.number_of_disjunctions == 0):
        cont_results = solve_continuous_problem(model, config)
        self.LB = cont_results.problem.lower_bound
        self.UB = cont_results.problem.upper_bound
        return self.pyomo_results
    util_block = self.original_util_block = add_util_block(model)
    add_disjunct_list(util_block)
    add_boolean_variable_lists(util_block)
    add_algebraic_variable_list(util_block)