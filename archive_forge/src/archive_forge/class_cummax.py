from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
class cummax(AxisAtom):
    """Cumulative maximum.
    """

    def __init__(self, x, axis: int=0) -> None:
        super(cummax, self).__init__(x, axis=axis)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest entry in x.
        """
        return np.maximum.accumulate(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
        """The same as the input.
        """
        return self.args[0].shape

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        value = np.array(value).ravel(order='F')
        maxes = np.maximum.accumulate(value)
        D = np.zeros((value.size, 1))
        D[0] = 1
        if value.size > 1:
            D[1:] = maxes[1:] > maxes[:-1]
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis]

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
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False