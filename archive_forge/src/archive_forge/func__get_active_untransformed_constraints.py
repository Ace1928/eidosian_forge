from math import fabs
from pyomo.contrib.gdpopt.solve_subproblem import solve_subproblem
from pyomo.contrib.gdpopt.util import fix_discrete_problem_solution_in_subproblem
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc
def _get_active_untransformed_constraints(self, util_block, config):
    """Yield constraints in disjuncts where the indicator value is set or
        fixed to True."""
    model = util_block.parent_block()
    for constr in util_block.global_constraint_list:
        yield constr
    for disj, constr_list in util_block.constraints_by_disjunct.items():
        if fabs(disj.binary_indicator_var.value - 1) <= config.integer_tolerance:
            for constr in constr_list:
                yield constr