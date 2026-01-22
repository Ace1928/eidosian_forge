import abc
from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
class AxisAtom(Atom):
    """
    An abstract base class for atoms that can be applied along an axis.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, expr, axis: Optional[int]=None, keepdims: bool=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super(AxisAtom, self).__init__(expr)

    def shape_from_args(self) -> Tuple[int, ...]:
        """Depends on axis.
        """
        shape = list(self.args[0].shape)
        if self.keepdims and self.axis is None:
            shape = [1] * len(shape)
        elif self.keepdims and self.axis is not None:
            shape[self.axis] = 1
        elif not self.keepdims and self.axis is None:
            shape = []
        else:
            shape = shape[:self.axis] + shape[self.axis + 1:]
        return tuple(shape)

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis, self.keepdims]

    def validate_arguments(self) -> None:
        """Checks that the new shape has the same number of entries as the old.
        """
        if self.axis is not None and self.axis > self.args[0].ndim:
            raise ValueError('Invalid argument for axis.')
        super(AxisAtom, self).validate_arguments()

    def _axis_grad(self, values) -> Optional[List[sp.csc_matrix]]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        Takes axis into account.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.axis is None or self.args[0].ndim < 2:
            value = np.reshape(values[0].T, (self.args[0].size, 1))
            D = self._column_grad(value)
            if D is not None:
                D = sp.csc_matrix(D)
        else:
            m, n = self.args[0].shape
            if self.axis == 0:
                D = sp.csc_matrix((m * n, n), dtype=float)
                for i in range(n):
                    value = values[0][:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    else:
                        d = np.array(d).flatten()
                    row = np.linspace(i * n, i * n + m - 1, m)
                    col = np.ones(m) * i
                    D = D + sp.csc_matrix((d, (row, col)), shape=(m * n, n))
            else:
                values = np.transpose(values[0])
                D = sp.csc_matrix((m * n, m), dtype=float)
                for i in range(m):
                    value = values[:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    row = np.linspace(i, i + (n - 1) * m, n)
                    col = np.ones(n) * i
                    D = D + sp.csc_matrix((np.array(d)[0], (row, col)), shape=(m * n, m))
        return [D]

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A SciPy sparse matrix or None.
        """
        raise NotImplementedError()