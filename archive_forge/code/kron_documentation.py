from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants.parameter import is_param_free
Kronecker product of two matrices.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        