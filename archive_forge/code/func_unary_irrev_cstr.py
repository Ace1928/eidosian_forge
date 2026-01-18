from .._util import get_backend
def unary_irrev_cstr(t, k, r, p, fr, fp, fv, backend=None):
    """Analytic solution for ``A -> B`` in a CSTR.

    Analytic solution for a first order process in a continuously
    stirred tank reactor (CSTR).

    Parameters
    ----------
    t : array_like
    k : float_like
        Rate constant
    r : float_like
        Initial concentration of reactant.
    p : float_like
        Initial concentration of product.
    fr : float_like
        Concentration of reactant in feed.
    fp : float_like
        Concentration of product in feed.
    fv : float_like
        Feed rate / tank volume ratio.
    backend : module or str
        Default is 'numpy', can also be e.g. ``sympy``.

    Returns
    -------
    length-2 tuple
        concentrations of reactant and product

    """
    be = get_backend(backend)
    x0 = fr * fv
    x1 = fv + k
    x2 = 1 / x1
    x3 = fv * r + k * r - x0
    x4 = fr * k
    x5 = be.exp(-fv * t)
    return (x0 * x2 + x2 * x3 * be.exp(-t * x1), -x2 * x3 * x5 * (-1 + be.exp(-k * t)) + x2 * x5 * (-fp * fv - fp * k + fv * p + k * p - x4) + x2 * (fp * x1 + x4))