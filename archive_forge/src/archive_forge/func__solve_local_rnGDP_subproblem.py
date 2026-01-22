from collections import namedtuple
from heapq import heappush, heappop
import traceback
from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, Constraint, TransformationFactory
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc
def _solve_local_rnGDP_subproblem(self, model, config):
    subproblem = TransformationFactory('gdp.bigm').create_using(model)
    obj_sense_correction = self.objective_sense != minimize
    subprob_utils = subproblem.component(self.original_util_block.name)
    model_utils = model.component(self.original_util_block.name)
    try:
        with SuppressInfeasibleWarning():
            result = SolverFactory(config.local_minlp_solver).solve(subproblem, **config.local_minlp_solver_args)
    except RuntimeError as e:
        config.logger.warning('Solver encountered RuntimeError. Treating as infeasible. Msg: %s\n%s' % (str(e), traceback.format_exc()))
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('-inf'), float('inf'))
    term_cond = result.solver.termination_condition
    if term_cond == tc.optimal:
        assert result.solver.status is SolverStatus.ok
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config)
        return (float('-inf'), ub)
    elif term_cond == tc.locallyOptimal or term_cond == tc.feasible:
        assert result.solver.status is SolverStatus.ok
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config)
        return (float('-inf'), ub)
    elif term_cond == tc.unbounded:
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('-inf'), float('-inf'))
    elif term_cond == tc.infeasible:
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('-inf'), float('inf'))
    else:
        config.logger.warning('Unknown termination condition of %s. Treating as infeasible.' % term_cond)
        copy_var_list_values(from_list=subprob_utils.algebraic_variable_list, to_list=model_utils.algebraic_variable_list, config=config, ignore_integrality=True)
        return (float('-inf'), float('inf'))