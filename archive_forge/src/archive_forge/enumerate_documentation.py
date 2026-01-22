from itertools import product
from pyomo.common.collections import ComponentSet
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.nlp_initialization import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.solve_subproblem import solve_subproblem
from pyomo.contrib.gdpopt.util import (
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc
from pyomo.opt.base import SolverFactory

    Solves Generalized Disjunctive Programming (GDP) by enumerating all
    discrete solutions and solving the resulting NLP subproblems, then
    returning the best solution found.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For non-convex problems,
    the algorithm will not be exact unless the NLP subproblems are solved
    globally.
    