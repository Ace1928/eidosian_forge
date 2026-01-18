import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08):
    """
    Compute a triple (definite) integral.

    Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
    ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.

    Parameters
    ----------
    func : function
        A Python function or method of at least three variables in the
        order (z, y, x).
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : function or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : function or float
        The upper boundary curve in y (same requirements as `gfun`).
    qfun : function or float
        The lower boundary surface in z.  It must be a function that takes
        two floats in the order (x, y) and returns a float or a float
        indicating a constant boundary surface.
    rfun : function or float
        The upper boundary surface in z. (Same requirements as `qfun`.)
    args : tuple, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the innermost 1-D quadrature
        integration. Default is 1.49e-8.
    epsrel : float, optional
        Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : Adaptive quadrature using QUADPACK
    quadrature : Adaptive Gaussian quadrature
    fixed_quad : Fixed-order Gaussian quadrature
    dblquad : Double integrals
    nquad : N-dimensional integrals
    romb : Integrators for sampled data
    simpson : Integrators for sampled data
    ode : ODE integrators
    odeint : ODE integrators
    scipy.special : For coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used, if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Compute the triple integral of ``x * y * z``, over ``x`` ranging
    from 1 to 2, ``y`` ranging from 2 to 3, ``z`` ranging from 0 to 1.
    That is, :math:`\\int^{x=2}_{x=1} \\int^{y=3}_{y=2} \\int^{z=1}_{z=0} x y z
    \\,dz \\,dy \\,dx`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> f = lambda z, y, x: x*y*z
    >>> integrate.tplquad(f, 1, 2, 2, 3, 0, 1)
    (1.8749999999999998, 3.3246447942574074e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=1-2x}_{y=0}
    \\int^{z=1-x-2y}_{z=0} x y z \\,dz \\,dy \\,dx`.
    Note: `qfun`/`rfun` takes arguments in the order (x, y), even though ``f``
    takes arguments in the order (z, y, x).

    >>> f = lambda z, y, x: x*y*z
    >>> integrate.tplquad(f, 0, 1, 0, lambda x: 1-2*x, 0, lambda x, y: 1-x-2*y)
    (0.05416666666666668, 2.1774196738157757e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=1}_{y=0} \\int^{z=1}_{z=0}
    a x y z \\,dz \\,dy \\,dx` for :math:`a=1, 3`.

    >>> f = lambda z, y, x, a: a*x*y*z
    >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(1,))
        (0.125, 5.527033708952211e-15)
    >>> integrate.tplquad(f, 0, 1, 0, 1, 0, 1, args=(3,))
        (0.375, 1.6581101126856635e-14)

    Compute the three-dimensional Gaussian Integral, which is the integral of
    the Gaussian function :math:`f(x,y,z) = e^{-(x^{2} + y^{2} + z^{2})}`, over
    :math:`(-\\infty,+\\infty)`. That is, compute the integral
    :math:`\\iiint^{+\\infty}_{-\\infty} e^{-(x^{2} + y^{2} + z^{2})} \\,dz
    \\,dy\\,dx`.

    >>> f = lambda x, y, z: np.exp(-(x ** 2 + y ** 2 + z ** 2))
    >>> integrate.tplquad(f, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
        (5.568327996830833, 4.4619078828029765e-08)

    """

    def ranges0(*args):
        return [qfun(args[1], args[0]) if callable(qfun) else qfun, rfun(args[1], args[0]) if callable(rfun) else rfun]

    def ranges1(*args):
        return [gfun(args[0]) if callable(gfun) else gfun, hfun(args[0]) if callable(hfun) else hfun]
    ranges = [ranges0, ranges1, [a, b]]
    return nquad(func, ranges, args=args, opts={'epsabs': epsabs, 'epsrel': epsrel})