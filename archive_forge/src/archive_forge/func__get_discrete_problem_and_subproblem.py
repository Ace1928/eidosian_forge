from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def _get_discrete_problem_and_subproblem(solver, config):
    util_block = solver.original_util_block
    original_model = util_block.parent_block()
    if config.force_subproblem_nlp:
        add_discrete_variable_list(util_block)
    original_obj = move_nonlinear_objective_to_constraints(util_block, config.logger)
    solver.original_obj = original_obj
    subproblem, subproblem_util_block = get_subproblem(original_model, util_block)
    start = get_main_elapsed_time(solver.timing)
    discrete_problem_util_block = initialize_discrete_problem(util_block, subproblem_util_block, config, solver)
    config.logger.info('Finished discrete problem initialization in {:.2f}s and {} iterations \n'.format(get_main_elapsed_time(solver.timing) - start, solver.initialization_iteration))
    return (discrete_problem_util_block, subproblem_util_block)