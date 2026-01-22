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
class TerminationCondition:
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """

    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, iter=None, norm=maxnorm):
        if f_tol is None:
            f_tol = np.finfo(np.float64).eps ** (1.0 / 3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.norm = norm
        self.iter = iter
        self.f0_norm = None
        self.iteration = 0

    def check(self, f, x, dx):
        self.iteration += 1
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)
        if self.f0_norm is None:
            self.f0_norm = f_norm
        if f_norm == 0:
            return 1
        if self.iter is not None:
            return 2 * (self.iteration > self.iter)
        return int((f_norm <= self.f_tol and f_norm / self.f_rtol <= self.f0_norm) and (dx_norm <= self.x_tol and dx_norm / self.x_rtol <= x_norm))