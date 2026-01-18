from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public
@public
def viete(f, roots=None, *gens, **args):
    """
    Generate Viete's formulas for ``f``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import viete
    >>> from sympy import symbols

    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')

    >>> viete(a*x**2 + b*x + c, [r1, r2], x)
    [(r1 + r2, -b/a), (r1*r2, c/a)]

    """
    allowed_flags(args, [])
    if isinstance(roots, Basic):
        gens, roots = ((roots,) + gens, None)
    try:
        f, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('viete', 1, exc)
    if f.is_multivariate:
        raise MultivariatePolynomialError('multivariate polynomials are not allowed')
    n = f.degree()
    if n < 1:
        raise ValueError("Cannot derive Viete's formulas for a constant polynomial")
    if roots is None:
        roots = numbered_symbols('r', start=1)
    roots = take(roots, n)
    if n != len(roots):
        raise ValueError('required %s roots, got %s' % (n, len(roots)))
    lc, coeffs = (f.LC(), f.all_coeffs())
    result, sign = ([], -1)
    for i, coeff in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign * (coeff / lc)
        result.append((poly, coeff))
        sign = -sign
    return result