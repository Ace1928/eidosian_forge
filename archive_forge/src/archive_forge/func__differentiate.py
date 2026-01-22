import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _differentiate(func, x, *, args=(), atol=None, rtol=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, callback=None):
    """Evaluate the derivative of an elementwise scalar function numerically.

    Parameters
    ----------
    func : callable
        The function whose derivative is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise function: each element
         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
    x : array_like
        Abscissae at which to evaluate the derivative.
    args : tuple, optional
        Additional positional arguments to be passed to `func`. Must be arrays
        broadcastable with `x`. If the callable to be differentiated requires
        arguments that are not broadcastable with `x`, wrap that callable with
        `func`. See Examples.
    atol, rtol : float, optional
        Absolute and relative tolerances for the stopping condition: iteration
        will stop when ``res.error < atol + rtol * abs(res.df)``. The default
        `atol` is the smallest normal number of the appropriate dtype, and
        the default `rtol` is the square root of the precision of the
        appropriate dtype.
    order : int, default: 8
        The (positive integer) order of the finite difference formula to be
        used. Odd integers will be rounded up to the next even integer.
    initial_step : float, default: 0.5
        The (absolute) initial step size for the finite difference derivative
        approximation.
    step_factor : float, default: 2.0
        The factor by which the step size is *reduced* in each iteration; i.e.
        the step size in iteration 1 is ``initial_step/step_factor``. If
        ``step_factor < 1``, subsequent steps will be greater than the initial
        step; this may be useful if steps smaller than some threshold are
        undesirable (e.g. due to subtractive cancellation error).
    maxiter : int, default: 10
        The maximum number of iterations of the algorithm to perform. See
        notes.
    step_direction : array_like
        An array representing the direction of the finite difference steps (for
        use when `x` lies near to the boundary of the domain of the function.)
        Must be broadcastable with `x` and all `args`.
        Where 0 (default), central differences are used; where negative (e.g.
        -1), steps are non-positive; and where positive (e.g. 1), all steps are
        non-negative.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is an ``OptimizeResult``
        similar to that returned by `_differentiate` (but containing the
        current iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_differentiate` will return a result.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. (The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.)

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The error estimate increased, so iteration was terminated.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        df : float
            The derivative of `func` at `x`, if the algorithm terminated
            successfully.
        error : float
            An estimate of the error: the magnitude of the difference between
            the current estimate of the derivative and the estimate in the
            previous iteration.
        nit : int
            The number of iterations performed.
        nfev : int
            The number of points at which `func` was evaluated.
        x : float
            The value at which the derivative of `func` was evaluated
            (after broadcasting with `args` and `step_direction`).

    Notes
    -----
    The implementation was inspired by jacobi [1]_, numdifftools [2]_, and
    DERIVEST [3]_, but the implementation follows the theory of Taylor series
    more straightforwardly (and arguably naively so).
    In the first iteration, the derivative is estimated using a finite
    difference formula of order `order` with maximum step size `initial_step`.
    Each subsequent iteration, the maximum step size is reduced by
    `step_factor`, and the derivative is estimated again until a termination
    condition is reached. The error estimate is the magnitude of the difference
    between the current derivative approximation and that of the previous
    iteration.

    The stencils of the finite difference formulae are designed such that
    abscissae are "nested": after `func` is evaluated at ``order + 1``
    points in the first iteration, `func` is evaluated at only two new points
    in each subsequent iteration; ``order - 1`` previously evaluated function
    values required by the finite difference formula are reused, and two
    function values (evaluations at the points furthest from `x`) are unused.

    Step sizes are absolute. When the step size is small relative to the
    magnitude of `x`, precision is lost; for example, if `x` is ``1e20``, the
    default initial step size of ``0.5`` cannot be resolved. Accordingly,
    consider using larger initial step sizes for large magnitudes of `x`.

    The default tolerances are challenging to satisfy at points where the
    true derivative is exactly zero. If the derivative may be exactly zero,
    consider specifying an absolute tolerance (e.g. ``atol=1e-16``) to
    improve convergence.

    References
    ----------
    [1]_ Hans Dembinski (@HDembinski). jacobi.
         https://github.com/HDembinski/jacobi
    [2]_ Per A. Brodtkorb and John D'Errico. numdifftools.
         https://numdifftools.readthedocs.io/en/latest/
    [3]_ John D'Errico. DERIVEST: Adaptive Robust Numerical Differentiation.
         https://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation
    [4]_ Numerical Differentition. Wikipedia.
         https://en.wikipedia.org/wiki/Numerical_differentiation

    Examples
    --------
    Evaluate the derivative of ``np.exp`` at several points ``x``.

    >>> import numpy as np
    >>> from scipy.optimize._zeros_py import _differentiate
    >>> f = np.exp
    >>> df = np.exp  # true derivative
    >>> x = np.linspace(1, 2, 5)
    >>> res = _differentiate(f, x)
    >>> res.df  # approximation of the derivative
    array([2.71828183, 3.49034296, 4.48168907, 5.75460268, 7.3890561 ])
    >>> res.error  # estimate of the error
    array(
        [7.12940817e-12, 9.16688947e-12, 1.17594823e-11, 1.50972568e-11, 1.93942640e-11]
    )
    >>> abs(res.df - df(x))  # true error
    array(
        [3.06421555e-14, 3.01980663e-14, 5.06261699e-14, 6.30606678e-14, 8.34887715e-14]
    )

    Show the convergence of the approximation as the step size is reduced.
    Each iteration, the step size is reduced by `step_factor`, so for
    sufficiently small initial step, each iteration reduces the error by a
    factor of ``1/step_factor**order`` until finite precision arithmetic
    inhibits further improvement.

    >>> iter = list(range(1, 12))  # maximum iterations
    >>> hfac = 2  # step size reduction per iteration
    >>> hdir = [-1, 0, 1]  # compare left-, central-, and right- steps
    >>> order = 4  # order of differentiation formula
    >>> x = 1
    >>> ref = df(x)
    >>> errors = []  # true error
    >>> for i in iter:
    ...     res = _differentiate(f, x, maxiter=i, step_factor=hfac,
    ...                          step_direction=hdir, order=order,
    ...                          atol=0, rtol=0)  # prevent early termination
    ...     errors.append(abs(res.df - ref))
    >>> errors = np.array(errors)
    >>> plt.semilogy(iter, errors[:, 0], label='left differences')
    >>> plt.semilogy(iter, errors[:, 1], label='central differences')
    >>> plt.semilogy(iter, errors[:, 2], label='right differences')
    >>> plt.xlabel('iteration')
    >>> plt.ylabel('error')
    >>> plt.legend()
    >>> plt.show()
    >>> (errors[1, 1] / errors[0, 1], 1 / hfac**order)
    (0.06215223140159822, 0.0625)

    The implementation is vectorized over `x`, `step_direction`, and `args`.
    The function is evaluated once before the first iteration to perform input
    validation and standardization, and once per iteration thereafter.

    >>> def f(x, p):
    ...     print('here')
    ...     f.nit += 1
    ...     return x**p
    >>> f.nit = 0
    >>> def df(x, p):
    ...     return p*x**(p-1)
    >>> x = np.arange(1, 5)
    >>> p = np.arange(1, 6).reshape((-1, 1))
    >>> hdir = np.arange(-1, 2).reshape((-1, 1, 1))
    >>> res = _differentiate(f, x, args=(p,), step_direction=hdir, maxiter=1)
    >>> np.allclose(res.df, df(x, p))
    True
    >>> res.df.shape
    (3, 5, 4)
    >>> f.nit
    2

    """
    res = _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step, step_factor, step_direction, callback)
    func, x, args, atol, rtol, maxiter, order, h0, fac, hdir, callback = res
    xs, fs, args, shape, dtype = _scalar_optimization_initialize(func, (x,), args)
    x, f = (xs[0], fs[0])
    df = np.full_like(f, np.nan)
    hdir = np.broadcast_to(hdir, shape).flatten()
    status = np.full_like(x, _EINPROGRESS, dtype=int)
    nit, nfev = (0, 1)
    il = hdir < 0
    ic = hdir == 0
    ir = hdir > 0
    io = il | ir
    work = OptimizeResult(x=x, df=df, fs=f[:, np.newaxis], error=np.nan, h=h0, df_last=np.nan, error_last=np.nan, h0=h0, fac=fac, atol=atol, rtol=rtol, nit=nit, nfev=nfev, status=status, dtype=dtype, terms=(order + 1) // 2, hdir=hdir, il=il, ic=ic, ir=ir, io=io)
    res_work_pairs = [('status', 'status'), ('df', 'df'), ('error', 'error'), ('nit', 'nit'), ('nfev', 'nfev'), ('x', 'x')]

    def pre_func_eval(work):
        """Determine the abscissae at which the function needs to be evaluated.

        See `_differentiate_weights` for a description of the stencil (pattern
        of the abscissae).

        In the first iteration, there is only one stored function value in
        `work.fs`, `f(x)`, so we need to evaluate at `order` new points. In
        subsequent iterations, we evaluate at two new points. Note that
        `work.x` is always flattened into a 1D array after broadcasting with
        all `args`, so we add a new axis at the end and evaluate all point
        in one call to the function.

        For improvement:
        - Consider measuring the step size actually taken, since `(x + h) - x`
          is not identically equal to `h` with floating point arithmetic.
        - Adjust the step size automatically if `x` is too big to resolve the
          step.
        - We could probably save some work if there are no central difference
          steps or no one-sided steps.
        """
        n = work.terms
        h = work.h
        c = work.fac
        d = c ** 0.5
        if work.nit == 0:
            hc = h / c ** np.arange(n)
            hc = np.concatenate((-hc[::-1], hc))
        else:
            hc = np.asarray([-h, h]) / c ** (n - 1)
        if work.nit == 0:
            hr = h / d ** np.arange(2 * n)
        else:
            hr = np.asarray([h, h / d]) / c ** (n - 1)
        n_new = 2 * n if work.nit == 0 else 2
        x_eval = np.zeros((len(work.hdir), n_new), dtype=work.dtype)
        il, ic, ir = (work.il, work.ic, work.ir)
        x_eval[ir] = work.x[ir, np.newaxis] + hr
        x_eval[ic] = work.x[ic, np.newaxis] + hc
        x_eval[il] = work.x[il, np.newaxis] - hr
        return x_eval

    def post_func_eval(x, f, work):
        """ Estimate the derivative and error from the function evaluations

        As in `pre_func_eval`: in the first iteration, there is only one stored
        function value in `work.fs`, `f(x)`, so we need to add the `order` new
        points. In subsequent iterations, we add two new points. The tricky
        part is getting the order to match that of the weights, which is
        described in `_differentiate_weights`.

        For improvement:
        - Change the order of the weights (and steps in `pre_func_eval`) to
          simplify `work_fc` concatenation and eliminate `fc` concatenation.
        - It would be simple to do one-step Richardson extrapolation with `df`
          and `df_last` to increase the order of the estimate and/or improve
          the error estimate.
        - Process the function evaluations in a more numerically favorable
          way. For instance, combining the pairs of central difference evals
          into a second-order approximation and using Richardson extrapolation
          to produce a higher order approximation seemed to retain accuracy up
          to very high order.
        - Alternatively, we could use `polyfit` like Jacobi. An advantage of
          fitting polynomial to more points than necessary is improved noise
          tolerance.
        """
        n = work.terms
        n_new = n if work.nit == 0 else 1
        il, ic, io = (work.il, work.ic, work.io)
        work_fc = (f[ic, :n_new], work.fs[ic, :], f[ic, -n_new:])
        work_fc = np.concatenate(work_fc, axis=-1)
        if work.nit == 0:
            fc = work_fc
        else:
            fc = (work_fc[:, :n], work_fc[:, n:n + 1], work_fc[:, -n:])
            fc = np.concatenate(fc, axis=-1)
        work_fo = np.concatenate((work.fs[io, :], f[io, :]), axis=-1)
        if work.nit == 0:
            fo = work_fo
        else:
            fo = np.concatenate((work_fo[:, 0:1], work_fo[:, -2 * n:]), axis=-1)
        work.fs = np.zeros((len(ic), work.fs.shape[-1] + 2 * n_new))
        work.fs[ic] = work_fc
        work.fs[io] = work_fo
        wc, wo = _differentiate_weights(work, n)
        work.df_last = work.df.copy()
        work.df[ic] = fc @ wc / work.h
        work.df[io] = fo @ wo / work.h
        work.df[il] *= -1
        work.h /= work.fac
        work.error_last = work.error
        work.error = abs(work.df - work.df_last)

    def check_termination(work):
        """Terminate due to convergence, non-finite values, or error increase"""
        stop = np.zeros_like(work.df).astype(bool)
        i = work.error < work.atol + work.rtol * abs(work.df)
        work.status[i] = _ECONVERGED
        stop[i] = True
        if work.nit > 0:
            i = ~(np.isfinite(work.x) & np.isfinite(work.df) | stop)
            work.df[i], work.status[i] = (np.nan, _EVALUEERR)
            stop[i] = True
        i = (work.error > work.error_last * 10) & ~stop
        work.status[i] = _EERRORINCREASE
        stop[i] = True
        return stop

    def post_termination_check(work):
        return

    def customize_result(res, shape):
        return shape
    return _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)