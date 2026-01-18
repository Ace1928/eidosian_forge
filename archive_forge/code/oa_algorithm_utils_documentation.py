from math import fabs
from pyomo.contrib.gdpopt.solve_subproblem import solve_subproblem
from pyomo.contrib.gdpopt.util import fix_discrete_problem_solution_in_subproblem
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc
Yield constraints in disjuncts where the indicator value is set or
        fixed to True.