from pyomo.common.config import document_kwargs_from_configdict, ConfigDict
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.config_options import (
from pyomo.opt.base import SolverFactory
def _handle_strategy_deprecation(config):
    if config.algorithm is None and config.strategy is not None:
        config.algorithm = config.strategy