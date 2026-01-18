from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdp_bounds.info import disjunctive_bounds
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.core import Constraint, Block, NonNegativeIntegers, Objective, value
from pyomo.core.expr.numvalue import is_potentially_variable
from pyomo.core.expr.visitor import identify_variables
from pyomo.opt.base import SolverFactory
Add affine cuts