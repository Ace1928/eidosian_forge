from functools import reduce
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
from sympy.polys.polytools import (
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
def _minpoly_compose(ex, x, dom):
    """
    Computes the minimal polynomial of an algebraic element
    using operations on minimal polynomials

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, Rational
    >>> from sympy.abc import x, y
    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=True)
    x**2 - 2*x - 1
    >>> minimal_polynomial(sqrt(y) + 1/y, x, compose=True)
    x**2*y**2 - 2*x*y - y**3 + 1

    """
    if ex.is_Rational:
        return ex.q * x - ex.p
    if ex is I:
        _, factors = factor_list(x ** 2 + 1, x, domain=dom)
        return x ** 2 + 1 if len(factors) == 1 else x - I
    if ex is S.GoldenRatio:
        _, factors = factor_list(x ** 2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x ** 2 - x - 1
        else:
            return _choose_factor(factors, x, (1 + sqrt(5)) / 2, dom=dom)
    if ex is S.TribonacciConstant:
        _, factors = factor_list(x ** 3 - x ** 2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x ** 3 - x ** 2 - x - 1
        else:
            fac = (1 + cbrt(19 - 3 * sqrt(33)) + cbrt(19 + 3 * sqrt(33))) / 3
            return _choose_factor(factors, x, fac, dom=dom)
    if hasattr(dom, 'symbols') and ex in dom.symbols:
        return x - ex
    if dom.is_QQ and _is_sum_surds(ex):
        ex -= x
        while 1:
            ex1 = _separate_sq(ex)
            if ex1 is ex:
                return ex
            else:
                ex = ex1
    if ex.is_Add:
        res = _minpoly_add(x, dom, *ex.args)
    elif ex.is_Mul:
        f = Factors(ex).factors
        r = sift(f.items(), lambda itx: itx[0].is_Rational and itx[1].is_Rational)
        if r[True] and dom == QQ:
            ex1 = Mul(*[bx ** ex for bx, ex in r[False] + r[None]])
            r1 = dict(r[True])
            dens = [y.q for y in r1.values()]
            lcmdens = reduce(lcm, dens, 1)
            neg1 = S.NegativeOne
            expn1 = r1.pop(neg1, S.Zero)
            nums = [base ** (y.p * lcmdens // y.q) for base, y in r1.items()]
            ex2 = Mul(*nums)
            mp1 = minimal_polynomial(ex1, x)
            mp2 = ex2.q * x ** lcmdens - ex2.p * neg1 ** (expn1 * lcmdens)
            ex2 = neg1 ** expn1 * ex2 ** Rational(1, lcmdens)
            res = _minpoly_op_algebraic_element(Mul, ex1, ex2, x, dom, mp1=mp1, mp2=mp2)
        else:
            res = _minpoly_mul(x, dom, *ex.args)
    elif ex.is_Pow:
        res = _minpoly_pow(ex.base, ex.exp, x, dom)
    elif ex.__class__ is sin:
        res = _minpoly_sin(ex, x)
    elif ex.__class__ is cos:
        res = _minpoly_cos(ex, x)
    elif ex.__class__ is tan:
        res = _minpoly_tan(ex, x)
    elif ex.__class__ is exp:
        res = _minpoly_exp(ex, x)
    elif ex.__class__ is CRootOf:
        res = _minpoly_rootof(ex, x)
    else:
        raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)
    return res