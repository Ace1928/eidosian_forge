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
def _create_pyomo_results_object_with_problem_info(self, original_model, config):
    """
        Initialize a results object with results.problem information
        """
    results = self.pyomo_results = SolverResults()
    results.solver.name = 'GDPopt %s - %s' % (self.version(), self.algorithm)
    prob = results.problem
    prob.name = original_model.name
    prob.number_of_nonzeros = None
    num_of = build_model_size_report(original_model)
    prob.number_of_constraints = num_of.activated.constraints
    prob.number_of_disjunctions = num_of.activated.disjunctions
    prob.number_of_variables = num_of.activated.variables
    prob.number_of_binary_variables = num_of.activated.binary_variables
    prob.number_of_continuous_variables = num_of.activated.continuous_variables
    prob.number_of_integer_variables = num_of.activated.integer_variables
    config.logger.info('Original model has %s constraints (%s nonlinear) and %s disjunctions, with %s variables, of which %s are binary, %s are integer, and %s are continuous.' % (num_of.activated.constraints, num_of.activated.nonlinear_constraints, num_of.activated.disjunctions, num_of.activated.variables, num_of.activated.binary_variables, num_of.activated.integer_variables, num_of.activated.continuous_variables))
    active_objectives = list(original_model.component_data_objects(ctype=Objective, active=True, descend_into=True))
    number_of_objectives = len(active_objectives)
    if number_of_objectives == 0:
        config.logger.warning('Model has no active objectives. Adding dummy objective.')
        self._dummy_obj = discrete_obj = Objective(expr=1)
        original_model.add_component(unique_component_name(original_model, 'dummy_obj'), discrete_obj)
    elif number_of_objectives > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        discrete_obj = active_objectives[0]
    prob.sense = minimize if discrete_obj.sense == 1 else maximize
    return results