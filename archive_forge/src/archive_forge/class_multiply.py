from __future__ import division
import operator as op
from functools import reduce
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import (
from cvxpy.expressions.expression import Expression
class multiply(MulExpression):
    """ Multiplies two expressions elementwise.
    """

    def __init__(self, lh_expr, rh_expr) -> None:
        lh_expr, rh_expr = self.broadcast(lh_expr, rh_expr)
        super(multiply, self).__init__(lh_expr, rh_expr)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_atom_quasiconvex(self) -> bool:
        return (self.args[0].is_constant() or self.args[1].is_constant()) or (self.args[0].is_nonneg() and self.args[1].is_nonpos()) or (self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_atom_quasiconcave(self) -> bool:
        return (self.args[0].is_constant() or self.args[1].is_constant()) or all((arg.is_nonneg() for arg in self.args)) or all((arg.is_nonpos() for arg in self.args))

    def numeric(self, values):
        """Multiplies the values elementwise.
        """
        if sp.issparse(values[0]):
            return values[0].multiply(values[1])
        elif sp.issparse(values[1]):
            return values[1].multiply(values[0])
        else:
            return np.multiply(values[0], values[1])

    def shape_from_args(self) -> Tuple[int, ...]:
        """The sum of the argument dimensions - 1.
        """
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        return self.args[0].is_psd() and self.args[1].is_psd() or (self.args[0].is_nsd() and self.args[1].is_nsd())

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        return self.args[0].is_psd() and self.args[1].is_nsd() or (self.args[0].is_nsd() and self.args[1].is_psd())

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the expressions elementwise.

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
            (LinOp for objective, list of exprraints)
        """
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.multiply(lhs, rhs), [])
        elif self.args[1].is_constant():
            return (lu.multiply(rhs, lhs), [])
        else:
            raise DCPError('Product of two non-constant expressions is not DCP.')