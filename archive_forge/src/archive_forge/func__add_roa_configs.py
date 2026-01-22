import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_roa_configs(CONFIG):
    """Adds the ROA-related configurations.

    Parameters
    ----------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy.
    """
    CONFIG.declare('level_coef', ConfigValue(default=0.5, domain=PositiveFloat, description='The coefficient in the regularization main problemrepresents how much the linear approximation of the MINLP problem is trusted.'))
    CONFIG.declare('solution_limit', ConfigValue(default=10, domain=PositiveInt, description='The solution limit for the regularization problem since it does not need to be solved to optimality.'))
    CONFIG.declare('sqp_lag_scaling_coef', ConfigValue(default='fixed', domain=In(['fixed', 'variable_dependent']), description='The coefficient used to scale the L2 norm in sqp_lag.'))