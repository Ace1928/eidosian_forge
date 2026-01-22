from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
def _get_algorithm_config():
    CONFIG = ConfigBlock('GDPoptAlgorithm')
    CONFIG.declare('strategy', ConfigValue(default=None, domain=_strategy_deprecation, description="DEPRECATED: Please use 'algorithm' instead."))
    CONFIG.declare('algorithm', ConfigValue(default=None, domain=In(_supported_algorithms), description='Algorithm to use.'))
    return CONFIG