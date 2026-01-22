from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public
def apart_undetermined_coeffs(P, Q):
    """Partial fractions via method of undetermined coefficients. """
    X = numbered_symbols(cls=Dummy)
    partial, symbols = ([], [])
    _, factors = Q.factor_list()
    for f, k in factors:
        n, q = (f.degree(), Q)
        for i in range(1, k + 1):
            coeffs, q = (take(X, n), q.quo(f))
            partial.append((coeffs, q, f, i))
            symbols.extend(coeffs)
    dom = Q.get_domain().inject(*symbols)
    F = Poly(0, Q.gen, domain=dom)
    for i, (coeffs, q, f, k) in enumerate(partial):
        h = Poly(coeffs, Q.gen, domain=dom)
        partial[i] = (h, f, k)
        q = q.set_domain(dom)
        F += h * q
    system, result = ([], S.Zero)
    for (k,), coeff in F.terms():
        system.append(coeff - P.nth(k))
    from sympy.solvers import solve
    solution = solve(system, symbols)
    for h, f, k in partial:
        h = h.as_expr().subs(solution)
        result += h / f.as_expr() ** k
    return result