import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _get_MindtPy_FP_config():
    """Set up the configurations for MindtPy-FP.

    Returns
    -------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy
    """
    CONFIG = ConfigBlock('MindtPy-GOA')
    CONFIG.declare('init_strategy', ConfigValue(default='FP', domain=In(['FP']), description='Initialization strategy', doc='Initialization strategy used by any method. Currently the continuous relaxation of the MINLP (rNLP), solve a maximal covering problem (max_binary), and fix the initial value for the integer variables (initial_binary).'))
    _add_common_configs(CONFIG)
    _add_fp_configs(CONFIG)
    _add_oa_cuts_configs(CONFIG)
    _add_subsolver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_bound_configs(CONFIG)
    return CONFIG