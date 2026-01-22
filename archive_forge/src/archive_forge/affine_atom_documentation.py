import abc
from typing import Any, List, Tuple
import scipy.sparse as sp
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants import Constant
from cvxpy.utilities import performance_utils as perf
Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        