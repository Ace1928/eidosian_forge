from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint
class diag_vec(AffAtom):
    """Converts a vector into a diagonal matrix.
    """

    def __init__(self, expr, k: int=0) -> None:
        self.k = k
        super(diag_vec, self).__init__(expr)

    def get_data(self) -> list[int]:
        return [self.k]

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        return np.diag(values[0], k=self.k)

    def shape_from_args(self) -> Tuple[int, int]:
        """A square matrix.
        """
        rows = self.args[0].shape[0] + abs(self.k)
        return (rows, rows)

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.k == 0

    def is_hermitian(self) -> bool:
        """Is the expression hermitian?
        """
        return self.k == 0

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        return self.is_nonneg() and self.k == 0

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        return self.is_nonpos() and self.k == 0

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Convolve two vectors.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.diag_vec(arg_objs[0], self.k), [])