from typing import List, Tuple
import numpy as np
from scipy.special import xlogy
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
class entr(Elementwise):
    """Elementwise :math:`-x\\log x`.
    """

    def __init__(self, x) -> None:
        super(entr, self).__init__(x)

    def numeric(self, values):
        x = values[0]
        results = -xlogy(x, x)
        if np.isscalar(results):
            if np.isnan(results):
                return -np.inf
        else:
            results[np.isnan(results)] = -np.inf
        return results

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size
        if np.min(values[0]) <= 0:
            return [None]
        else:
            grad_vals = -np.log(values[0]) - 1
            return [entr.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0]