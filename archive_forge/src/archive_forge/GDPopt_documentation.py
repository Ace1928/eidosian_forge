from pyomo.common.config import document_kwargs_from_configdict, ConfigDict
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.config_options import (
from pyomo.opt.base import SolverFactory
Return a 3-tuple describing the solver version.