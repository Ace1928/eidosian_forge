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
class DivExpression(BinaryOperator):
    """Division by scalar.

    Can be created by using the / operator of expression.
    """
    OP_NAME = '/'
    OP_FUNC = np.divide

    def __init__(self, lh_expr, rh_expr) -> None:
        lh_expr, rh_expr = self.broadcast(lh_expr, rh_expr)
        super(DivExpression, self).__init__(lh_expr, rh_expr)

    def numeric(self, values):
        """Divides numerator by denominator.
        """
        for i in range(2):
            if sp.issparse(values[i]):
                values[i] = values[i].todense().A
        return np.divide(values[0], values[1])

    def is_quadratic(self) -> bool:
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        """Can be a quadratic term if divisor is constant."""
        return self.args[0].has_quadratic_term() and self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        return self.args[0].is_qpwa() and self.args[1].is_constant()

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def is_atom_convex(self) -> bool:
        """Division is convex (affine) in its arguments only if
           the denominator is constant.
        """
        return self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_atom_quasiconvex(self) -> bool:
        return self.args[1].is_nonneg() or self.args[1].is_nonpos()

    def is_atom_quasiconcave(self) -> bool:
        return self.is_atom_quasiconvex()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonneg()
        else:
            return self.args[0].is_nonpos()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonpos()
        else:
            return self.args[0].is_nonneg()

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the linear expressions.

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
        return (lu.div_expr(arg_objs[0], arg_objs[1]), [])