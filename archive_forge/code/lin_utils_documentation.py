from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
Replaces parameters with constant nodes.

    Parameters
    ----------
    expr : LinOp
        The expression to replace parameters in.

    Returns
    -------
    LinOp
        An LinOp identical to expr, but with the parameters replaced.
    