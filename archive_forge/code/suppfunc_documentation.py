from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable

        Parameters
        ----------
        y : cvxpy.expressions.expression.Expression
            Must satisfy ``y.is_affine() == True``.

        parent : cvxpy.transforms.suppfunc.SuppFunc
            The object containing data for the convex set associated with this atom.
        