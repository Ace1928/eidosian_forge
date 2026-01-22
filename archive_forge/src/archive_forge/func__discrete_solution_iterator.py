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
def _discrete_solution_iterator(self, disjunctions, non_indicator_boolean_vars, discrete_var_list, config):
    discrete_var_values = [range(v.lb, v.ub + 1) for v in discrete_var_list]
    for true_indicators in product(*[disjunction.disjuncts for disjunction in disjunctions]):
        if not config.force_subproblem_nlp:
            yield (ComponentSet(true_indicators), (), ())
        else:
            for boolean_realization in product([True, False], repeat=len(non_indicator_boolean_vars)):
                for integer_realization in product(*discrete_var_values):
                    yield (ComponentSet(true_indicators), boolean_realization, integer_realization)