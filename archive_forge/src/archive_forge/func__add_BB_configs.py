from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
def _add_BB_configs(CONFIG):
    CONFIG.declare('check_sat', ConfigValue(default=False, domain=bool, description='\n            When True, GDPopt-LBB will check satisfiability\n            at each node via the pyomo.contrib.satsolver interface'))
    CONFIG.declare('solve_local_rnGDP', ConfigValue(default=False, domain=bool, description='\n            When True, GDPopt-LBB will solve a local MINLP at each node.'))