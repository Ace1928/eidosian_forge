import sys
from typing import Any, List, Tuple
import numpy as np
from cvxpy.atoms.elementwise.elementwise import Elementwise
class minimum(Elementwise):
    """Elementwise minimum of a sequence of expressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        """Requires at least 2 arguments.
        """
        super(minimum, self).__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise maximum.
        """
        return reduce(np.minimum, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        is_pos = all((arg.is_nonneg() for arg in self.args))
        is_neg = any((arg.is_nonpos() for arg in self.args))
        return (is_pos, is_neg)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return all((arg.is_pwl() for arg in self.args))

    def _grad(self, values) -> List[Any]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        min_vals = np.array(self.numeric(values))
        unused = np.array(np.ones(min_vals.shape), dtype=bool)
        grad_list = []
        for idx, value in enumerate(values):
            rows = self.args[idx].size
            cols = self.size
            grad_vals = (value == min_vals) & unused
            unused[value == min_vals] = 0
            grad_list += [minimum.elemwise_grad_to_diag(grad_vals, rows, cols)]
        return grad_list