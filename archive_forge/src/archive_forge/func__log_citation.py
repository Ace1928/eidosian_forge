from collections import namedtuple
from math import copysign
from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import (
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct
from pyomo.opt.base import SolverFactory
from pyomo.repn import generate_standard_repn
def _log_citation(self, config):
    config.logger.info('\n' + '- LOA algorithm:\n        Türkay, M; Grossmann, IE.\n        Logic-based MINLP algorithms for the optimal synthesis of process\n        networks. Comp. and Chem. Eng. 1996, 20(8), 959–978.\n        DOI: 10.1016/0098-1354(95)00219-7.\n        '.strip())