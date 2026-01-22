from __future__ import division
from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import rel_entr as rel_entr_scipy
from cvxpy.atoms.elementwise.elementwise import Elementwise
class rel_entr(Elementwise):
    """:math:`x\\log(x/y)`

    For disambiguation between rel_entr and kl_div, see https://github.com/cvxpy/cvxpy/issues/733
    """

    def __init__(self, x, y) -> None:
        super(rel_entr, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        return rel_entr_scipy(x, y)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        if idx == 0:
            return False
        else:
            return True

    def _grad(self, values) -> List[Optional[csc_matrix]]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if np.min(values[0]) <= 0 or np.min(values[1]) <= 0:
            return [None, None]
        else:
            div = values[0] / values[1]
            grad_vals = [np.log(div) + 1, -div]
            grad_list = []
            for idx in range(len(values)):
                rows = self.args[idx].size
                cols = self.size
                grad_list += [rel_entr.elemwise_grad_to_diag(grad_vals[idx], rows, cols)]
            return grad_list

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0, self.args[1] >= 0]