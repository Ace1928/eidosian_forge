from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public
@xthreaded
@public
def apart(f, x=None, full=False, **options):
    """
    Compute partial fraction decomposition of a rational function.

    Given a rational function ``f``, computes the partial fraction
    decomposition of ``f``. Two algorithms are available: One is based on the
    undertermined coefficients method, the other is Bronstein's full partial
    fraction decomposition algorithm.

    The undetermined coefficients method (selected by ``full=False``) uses
    polynomial factorization (and therefore accepts the same options as
    factor) for the denominator. Per default it works over the rational
    numbers, therefore decomposition of denominators with non-rational roots
    (e.g. irrational, complex roots) is not supported by default (see options
    of factor).

    Bronstein's algorithm can be selected by using ``full=True`` and allows a
    decomposition of denominators with non-rational roots. A human-readable
    result can be obtained via ``doit()`` (see examples below).

    Examples
    ========

    >>> from sympy.polys.partfrac import apart
    >>> from sympy.abc import x, y

    By default, using the undetermined coefficients method:

    >>> apart(y/(x + 2)/(x + 1), x)
    -y/(x + 2) + y/(x + 1)

    The undetermined coefficients method does not provide a result when the
    denominators roots are not rational:

    >>> apart(y/(x**2 + x + 1), x)
    y/(x**2 + x + 1)

    You can choose Bronstein's algorithm by setting ``full=True``:

    >>> apart(y/(x**2 + x + 1), x, full=True)
    RootSum(_w**2 + _w + 1, Lambda(_a, (-2*_a*y/3 - y/3)/(-_a + x)))

    Calling ``doit()`` yields a human-readable result:

    >>> apart(y/(x**2 + x + 1), x, full=True).doit()
    (-y/3 - 2*y*(-1/2 - sqrt(3)*I/2)/3)/(x + 1/2 + sqrt(3)*I/2) + (-y/3 -
        2*y*(-1/2 + sqrt(3)*I/2)/3)/(x + 1/2 - sqrt(3)*I/2)


    See Also
    ========

    apart_list, assemble_partfrac_list
    """
    allowed_flags(options, [])
    f = sympify(f)
    if f.is_Atom:
        return f
    else:
        P, Q = f.as_numer_denom()
    _options = options.copy()
    options = set_defaults(options, extension=True)
    try:
        (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)
    except PolynomialError as msg:
        if f.is_commutative:
            raise PolynomialError(msg)
        if f.is_Mul:
            c, nc = f.args_cnc(split_1=False)
            nc = f.func(*nc)
            if c:
                c = apart(f.func._from_args(c), x=x, full=full, **_options)
                return c * nc
            else:
                return nc
        elif f.is_Add:
            c = []
            nc = []
            for i in f.args:
                if i.is_commutative:
                    c.append(i)
                else:
                    try:
                        nc.append(apart(i, x=x, full=full, **_options))
                    except NotImplementedError:
                        nc.append(i)
            return apart(f.func(*c), x=x, full=full, **_options) + f.func(*nc)
        else:
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                try:
                    reps.append((e, apart(e, x=x, full=full, **_options)))
                    pot.skip()
                except NotImplementedError:
                    pass
            return f.xreplace(dict(reps))
    if P.is_multivariate:
        fc = f.cancel()
        if fc != f:
            return apart(fc, x=x, full=full, **_options)
        raise NotImplementedError('multivariate partial fraction decomposition')
    common, P, Q = P.cancel(Q)
    poly, P = P.div(Q, auto=True)
    P, Q = P.rat_clear_denoms(Q)
    if Q.degree() <= 1:
        partial = P / Q
    elif not full:
        partial = apart_undetermined_coeffs(P, Q)
    else:
        partial = apart_full_decomposition(P, Q)
    terms = S.Zero
    for term in Add.make_args(partial):
        if term.has(RootSum):
            terms += term
        else:
            terms += factor(term)
    return common * (poly.as_expr() + terms)