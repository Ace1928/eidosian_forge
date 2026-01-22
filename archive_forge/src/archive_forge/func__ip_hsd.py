import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args):
    """
    Solve a linear programming problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    using the interior point method of [4].

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A`` (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    alpha0 : float
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\\beta_3`of [4] Table 8.1
    beta : float
        The desired reduction of the path parameter :math:`\\mu` (see  [6]_)
    maxiter : int
        The maximum number of iterations of the algorithm.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    tol : float
        Termination tolerance; see [4]_ Section 4.5.
    sparse : bool
        Set to ``True`` if the problem is to be treated as sparse. However,
        the inputs ``A_eq`` and ``A_ub`` should nonetheless be provided as
        (dense) arrays rather than sparse matrices.
    lstsq : bool
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left as ``False`` unless severe
        numerical difficulties are frequently encountered, and a better option
        would be to improve the formulation of the problem.
    sym_pos : bool
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix (almost always).
    cholesky : bool
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for moderate, dense problems
        that are numerically well-behaved.
    pc : bool
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. It's unclear whether this is beneficial.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed. This is always
                1 for the interior-point method because it has only one phase.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Returns
    -------
    x_hat : float
        Solution vector (for standard form problem).
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf

    """
    iteration = 0
    x, y, z, tau, kappa = _get_blind_start(A.shape)
    ip = ip if pc else False
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol
    if disp:
        _display_iter(rho_p, rho_d, rho_g, '-', rho_mu, obj, header=True)
    if callback is not None:
        x_o, fun, slack, con = _postsolve(x / tau, postsolve_args)
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack, 'con': con, 'nit': iteration, 'phase': 1, 'complete': False, 'status': 0, 'message': '', 'success': False})
        callback(res)
    status = 0
    message = 'Optimization terminated successfully.'
    if sparse:
        A = sps.csc_matrix(A)
    while go:
        iteration += 1
        if ip:
            gamma = 1

            def eta(g):
                return 1
        else:
            gamma = 0 if pc else beta * np.mean(z * x)

            def eta(g=gamma):
                return 1 - g
        try:
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec)
            if ip:
                alpha = 1.0
                x, y, z, tau, kappa = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False
            else:
                alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)
                x, y, z, tau, kappa = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)
        except (LinAlgError, FloatingPointError, ValueError, ZeroDivisionError):
            status = 4
            message = _get_message(status)
            break
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(A, b, c, c0, x, y, z, tau, kappa)
        go = rho_p > tol or rho_d > tol or rho_A > tol
        if disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)
        if callback is not None:
            x_o, fun, slack, con = _postsolve(x / tau, postsolve_args)
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack, 'con': con, 'nit': iteration, 'phase': 1, 'complete': False, 'status': 0, 'message': '', 'success': False})
            callback(res)
        inf1 = rho_p < tol and rho_d < tol and (rho_g < tol) and (tau < tol * max(1, kappa))
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            if b.transpose().dot(y) > tol:
                status = 2
            else:
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break
    x_hat = x / tau
    return (x_hat, status, message, iteration)