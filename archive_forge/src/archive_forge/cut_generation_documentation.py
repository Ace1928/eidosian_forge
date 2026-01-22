from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
Adds affine cuts using MCPP.

    Parameters
    ----------
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    