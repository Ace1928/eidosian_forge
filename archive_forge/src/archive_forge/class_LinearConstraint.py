import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
class LinearConstraint:
    """Linear constraint on the variables.

    The constraint has the general inequality form::

        lb <= A.dot(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    A : {array_like, sparse matrix}, shape (m, n)
        Matrix defining the constraint.
    lb, ub : dense array_like, optional
        Lower and upper limits on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no limits).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """

    def _input_validation(self):
        if self.A.ndim != 2:
            message = '`A` must have exactly two dimensions.'
            raise ValueError(message)
        try:
            shape = self.A.shape[0:1]
            self.lb = np.broadcast_to(self.lb, shape)
            self.ub = np.broadcast_to(self.ub, shape)
            self.keep_feasible = np.broadcast_to(self.keep_feasible, shape)
        except ValueError:
            message = '`lb`, `ub`, and `keep_feasible` must be broadcastable to shape `A.shape[0:1]`'
            raise ValueError(message)

    def __init__(self, A, lb=-np.inf, ub=np.inf, keep_feasible=False):
        if not issparse(A):
            with catch_warnings():
                simplefilter('error')
                self.A = np.atleast_2d(A).astype(np.float64)
        else:
            self.A = A
        if issparse(lb) or issparse(ub):
            raise ValueError('Constraint limits must be dense arrays.')
        self.lb = np.atleast_1d(lb).astype(np.float64)
        self.ub = np.atleast_1d(ub).astype(np.float64)
        if issparse(keep_feasible):
            raise ValueError('`keep_feasible` must be a dense array.')
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def residual(self, x):
        """
        Calculate the residual between the constraint function and the limits

        For a linear constraint of the form::

            lb <= A@x <= ub

        the lower and upper residuals between ``A@x`` and the limits are values
        ``sl`` and ``sb`` such that::

            lb + sl == A@x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        the constraint are satisfied; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of the constraint is not
        satisfied.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return (self.A @ x - self.lb, self.ub - self.A @ x)