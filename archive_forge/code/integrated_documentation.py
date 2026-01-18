from .._util import get_backend
Analytic solution for ``2 A -> n B`` in a CSTR.

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
    n : int
    backend : module or str
        Default is 'numpy', can also be e.g. ``sympy``.

    Returns
    -------
    length-2 tuple
        concentrations of reactant and product

    