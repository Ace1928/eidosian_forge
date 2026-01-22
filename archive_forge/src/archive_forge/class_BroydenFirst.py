import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class BroydenFirst(GenericBroyden):
    """
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \\"Broyden's good method\\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden1'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) dx^\\dagger H / ( dx^\\dagger H df)

    which corresponds to Broyden's first Jacobian update

    .. math:: J_+ = J + (df - J dx) dx^\\dagger / dx^\\dagger dx


    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \\"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden1(fun, [0, 0])
    >>> sol
    array([0.84116396, 0.15883641])

    """

    def __init__(self, alpha=None, reduction_method='restart', max_rank=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.Gm = None
        if max_rank is None:
            max_rank = np.inf
        self.max_rank = max_rank
        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        reduce_params = (max_rank - 1,) + reduce_params
        if reduction_method == 'svd':
            self._reduce = lambda: self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda: self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda: self.Gm.restart_reduce(*reduce_params)
        else:
            raise ValueError("Unknown rank reduction method '%s'" % reduction_method)

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        return inv(self.Gm)

    def solve(self, f, tol=0):
        r = self.Gm.matvec(f)
        if not np.isfinite(r).all():
            self.setup(self.last_x, self.last_f, self.func)
            return self.Gm.matvec(f)
        return r

    def matvec(self, f):
        return self.Gm.solve(f)

    def rsolve(self, f, tol=0):
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()
        v = self.Gm.rmatvec(dx)
        c = dx - self.Gm.matvec(df)
        d = v / vdot(df, v)
        self.Gm.append(c, d)