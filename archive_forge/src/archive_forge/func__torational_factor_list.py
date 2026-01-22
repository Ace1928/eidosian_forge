from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
def _torational_factor_list(p, x):
    """
    helper function to factor polynomial using to_rational_coeffs

    Examples
    ========

    >>> from sympy.polys.polytools import _torational_factor_list
    >>> from sympy.abc import x
    >>> from sympy import sqrt, expand, Mul
    >>> p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}))
    >>> factors = _torational_factor_list(p, x); factors
    (-2, [(-x*(1 + sqrt(2))/2 + 1, 1), (-x*(1 + sqrt(2)) - 1, 1), (-x*(1 + sqrt(2)) + 1, 1)])
    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p
    True
    >>> p = expand(((x**2-1)*(x-2)).subs({x:x + sqrt(2)}))
    >>> factors = _torational_factor_list(p, x); factors
    (1, [(x - 2 + sqrt(2), 1), (x - 1 + sqrt(2), 1), (x + 1 + sqrt(2), 1)])
    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p
    True

    """
    from sympy.simplify.simplify import simplify
    p1 = Poly(p, x, domain='EX')
    n = p1.degree()
    res = to_rational_coeffs(p1)
    if not res:
        return None
    lc, r, t, g = res
    factors = factor_list(g.as_expr())
    if lc:
        c = simplify(factors[0] * lc * r ** n)
        r1 = simplify(1 / r)
        a = []
        for z in factors[1:][0]:
            a.append((simplify(z[0].subs({x: x * r1})), z[1]))
    else:
        c = factors[0]
        a = []
        for z in factors[1:][0]:
            a.append((z[0].subs({x: x - t}), z[1]))
    return (c, a)