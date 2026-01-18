from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
def update_set_covering_objective(discrete_problem_util_block, disj_needs_cover):
    num_needs_cover = sum((1 for disj_bool in disj_needs_cover if disj_bool))
    num_covered = len(disj_needs_cover) - num_needs_cover
    weights = list((num_covered + 1 if disj_bool else 1 for disj_bool in disj_needs_cover))
    discrete_problem_util_block.set_cover_obj.expr = sum((weight * disj.binary_indicator_var for weight, disj in zip(weights, discrete_problem_util_block.disjunct_list)))