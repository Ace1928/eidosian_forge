import numpy as np
from warnings import warn
from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin

    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make. If more are needed to
        meet convergence, `NoConvergence` is raised.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        rdiff : float, optional
            Relative step size to use in numerical differentiation.
        method : str or callable, optional
            Krylov method to use to approximate the Jacobian.  Can be a string,
            or a function implementing the same interface as the iterative
            solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
            ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
            ``'tfqmr'``.

            The default is `scipy.sparse.linalg.lgmres`.
        inner_M : LinearOperator or InverseJacobian
            Preconditioner for the inner Krylov iteration.
            Note that you can use also inverse Jacobians as (adaptive)
            preconditioners. For example,

            >>> jac = BroydenFirst()
            >>> kjac = KrylovJacobian(inner_M=jac.inverse).

            If the preconditioner has a method named 'update', it will
            be called as ``update(x, f)`` after each nonlinear step,
            with ``x`` giving the current point, and ``f`` the current
            function value.
        inner_tol, inner_maxiter, ...
            Parameters to pass on to the "inner" Krylov solver.
            See `scipy.sparse.linalg.gmres` for details.
        outer_k : int, optional
            Size of the subspace kept across LGMRES nonlinear
            iterations.

            See `scipy.sparse.linalg.lgmres` for details.
    