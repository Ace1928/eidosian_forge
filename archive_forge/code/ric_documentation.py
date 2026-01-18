from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import time_code
from pyomo.core import Objective
from pyomo.opt.base import SolverFactory
The GDPopt (Generalized Disjunctive Programming optimizer) relaxation
    with integer cuts (RIC) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For non-convex problems, RIC
    will not be exact unless the NLP subproblems are solved globally.
    