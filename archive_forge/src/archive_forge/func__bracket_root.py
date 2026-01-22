import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _bracket_root(func, a, b=None, *, min=None, max=None, factor=None, args=(), maxiter=1000):
    """Bracket the root of a monotonic scalar function of one variable

    This function works elementwise when `a`, `b`, `min`, `max`, `factor`, and
    the elements of `args` are broadcastable arrays.

    Parameters
    ----------
    func : callable
        The function for which the root is to be bracketed.
        The signature must be::

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with `x`. ``func`` must be an elementwise function: each element
        ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
    a, b : float array_like
        Starting guess of bracket, which need not contain a root. If `b` is
        not provided, ``b = a + 1``. Must be broadcastable with one another.
    min, max : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with `a` and `b`.
    factor : float array_like, default: 2
        The factor used to grow the bracket. See notes for details.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `a`, `b`, `min`, and `max`. If the callable to be
        bracketed requires arguments that are not broadcastable with these
        arrays, wrap that callable with `func` such that `func` accepts
        only `x` and broadcastable arrays.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.

    Returns
    -------
    res : OptimizeResult
        An instance of `scipy.optimize.OptimizeResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        xl, xr : float
            The lower and upper ends of the bracket, if the algorithm
            terminated successfully.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.
        nfev : int
            The number of function evaluations required to find the bracket.
            This is distinct from the number of times `func` is *called*
            because the function may evaluated at multiple points in a single
            call.
        nit : int
            The number of iterations of the algorithm that were performed.
        status : int
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits without finding a bracket.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).
            - ``2`` : A bracket was found in the opposite search direction (in `callback` only).

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).

    Notes
    -----
    This function generalizes an algorithm found in pieces throughout
    `scipy.stats`. The strategy is to iteratively grow the bracket `(l, r)`
     until ``func(l) < 0 < func(r)``. The bracket grows to the left as follows.

    - If `min` is not provided, the distance between `b` and `l` is iteratively
      increased by `factor`.
    - If `min` is provided, the distance between `min` and `l` is iteratively
      decreased by `factor`. Note that this also *increases* the bracket size.

    Growth of the bracket to the right is analogous.

    Growth of the bracket in one direction stops when the endpoint is no longer
    finite, the function value at the endpoint is no longer finite, or the
    endpoint reaches its limiting value (`min` or `max`). Iteration terminates
    when the bracket stops growing in both directions, the bracket surrounds
    the root, or a root is found (accidentally).

    If two brackets are found - that is, a bracket is found on both sides in
    the same iteration, the smaller of the two is returned.
    If roots of the function are found, both `l` and `r` are set to the
    leftmost root.

    """
    callback = None
    temp = _bracket_root_iv(func, a, b, min, max, factor, args, maxiter)
    func, a, b, min, max, factor, args, maxiter = temp
    xs = (a, b)
    temp = _scalar_optimization_initialize(func, xs, args)
    xs, fs, args, shape, dtype = temp
    x = np.concatenate(xs)
    f = np.concatenate(fs)
    n = len(x) // 2
    x_last = np.concatenate((x[n:], x[:n]))
    f_last = np.concatenate((f[n:], f[:n]))
    x0 = x_last
    min = np.broadcast_to(min, shape).astype(dtype, copy=False).ravel()
    max = np.broadcast_to(max, shape).astype(dtype, copy=False).ravel()
    limit = np.concatenate((min, max))
    factor = np.broadcast_to(factor, shape).astype(dtype, copy=False).ravel()
    factor = np.concatenate((factor, factor))
    active = np.arange(2 * n)
    args = [np.concatenate((arg, arg)) for arg in args]
    shape = shape + (2,)
    i = np.isinf(limit)
    ni = ~i
    d = np.zeros_like(x)
    d[i] = x[i] - x0[i]
    d[ni] = limit[ni] - x[ni]
    status = np.full_like(x, _EINPROGRESS, dtype=int)
    nit, nfev = (0, 1)
    work = OptimizeResult(x=x, x0=x0, f=f, limit=limit, factor=factor, active=active, d=d, x_last=x_last, f_last=f_last, nit=nit, nfev=nfev, status=status, args=args, xl=None, xr=None, fl=None, fr=None, n=n)
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xr', 'xr'), ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'), ('fr', 'fr'), ('x', 'x'), ('f', 'f'), ('x_last', 'x_last'), ('f_last', 'f_last')]

    def pre_func_eval(work):
        x = np.zeros_like(work.x)
        i = np.isinf(work.limit)
        work.d[i] *= work.factor[i]
        x[i] = work.x0[i] + work.d[i]
        ni = ~i
        work.d[ni] /= work.factor[ni]
        x[ni] = work.limit[ni] - work.d[ni]
        return x

    def post_func_eval(x, f, work):
        work.x_last = work.x
        work.f_last = work.f
        work.x = x
        work.f = f

    def check_termination(work):
        stop = np.zeros_like(work.x, dtype=bool)
        sf = np.sign(work.f)
        sf_last = np.sign(work.f_last)
        i = (sf_last == -sf) | (sf_last == 0) | (sf == 0)
        work.status[i] = _ECONVERGED
        stop[i] = True
        also_stop = (work.active[i] + work.n) % (2 * work.n)
        j = np.searchsorted(work.active, also_stop)
        j = j[j < len(work.active)]
        j = j[also_stop == work.active[j]]
        i = np.zeros_like(stop)
        i[j] = True
        i = i & ~stop
        work.status[i] = _ESTOPONESIDE
        stop[i] = True
        i = (work.x == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True
        i = ~(np.isfinite(work.x) & np.isfinite(work.f)) & ~stop
        work.status[i] = _EVALUEERR
        stop[i] = True
        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        n = len(res['x']) // 2
        xal = res['x'][:n]
        xar = res['x_last'][:n]
        xbl = res['x_last'][n:]
        xbr = res['x'][n:]
        fal = res['f'][:n]
        far = res['f_last'][:n]
        fbl = res['f_last'][n:]
        fbr = res['f'][n:]
        xl = xal.copy()
        fl = fal.copy()
        xr = xbr.copy()
        fr = fbr.copy()
        sa = res['status'][:n]
        sb = res['status'][n:]
        da = xar - xal
        db = xbr - xbl
        i1 = (da <= db) & (sa == 0) | (sa == 0) & (sb != 0)
        i2 = (db <= da) & (sb == 0) | (sb == 0) & (sa != 0)
        xr[i1] = xar[i1]
        fr[i1] = far[i1]
        xl[i2] = xbl[i2]
        fl[i2] = fbl[i2]
        res['xl'] = xl
        res['xr'] = xr
        res['fl'] = fl
        res['fr'] = fr
        res['nit'] = np.maximum(res['nit'][:n], res['nit'][n:])
        res['nfev'] = res['nfev'][:n] + res['nfev'][n:]
        res['status'] = np.choose(sa == 0, (sb, sa))
        res['success'] = res['status'] == 0
        del res['x']
        del res['f']
        del res['x_last']
        del res['f_last']
        return shape[:-1]
    return _scalar_optimization_loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval, post_func_eval, check_termination, post_termination_check, customize_result, res_work_pairs)