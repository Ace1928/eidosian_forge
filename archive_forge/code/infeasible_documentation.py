from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
Print the variables and constraints that are near their bounds.

    See :py:func:`find_close_to_bounds()` for a description of the
    variables and constraints that are returned (and which are omitted).

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    