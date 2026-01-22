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
def _parallel_poly_from_expr(exprs, opt):
    """Construct polynomials from expressions. """
    if len(exprs) == 2:
        f, g = exprs
        if isinstance(f, Poly) and isinstance(g, Poly):
            f = f.__class__._from_poly(f, opt)
            g = g.__class__._from_poly(g, opt)
            f, g = f.unify(g)
            opt.gens = f.gens
            opt.domain = f.domain
            if opt.polys is None:
                opt.polys = True
            return ([f, g], opt)
    origs, exprs = (list(exprs), [])
    _exprs, _polys = ([], [])
    failed = False
    for i, expr in enumerate(origs):
        expr = sympify(expr)
        if isinstance(expr, Basic):
            if expr.is_Poly:
                _polys.append(i)
            else:
                _exprs.append(i)
                if opt.expand:
                    expr = expr.expand()
        else:
            failed = True
        exprs.append(expr)
    if failed:
        raise PolificationFailed(opt, origs, exprs, True)
    if _polys:
        for i in _polys:
            exprs[i] = exprs[i].as_expr()
    reps, opt = _parallel_dict_from_expr(exprs, opt)
    if not opt.gens:
        raise PolificationFailed(opt, origs, exprs, True)
    from sympy.functions.elementary.piecewise import Piecewise
    for k in opt.gens:
        if isinstance(k, Piecewise):
            raise PolynomialError('Piecewise generators do not make sense')
    coeffs_list, lengths = ([], [])
    all_monoms = []
    all_coeffs = []
    for rep in reps:
        monoms, coeffs = list(zip(*list(rep.items())))
        coeffs_list.extend(coeffs)
        all_monoms.append(monoms)
        lengths.append(len(coeffs))
    domain = opt.domain
    if domain is None:
        opt.domain, coeffs_list = construct_domain(coeffs_list, opt=opt)
    else:
        coeffs_list = list(map(domain.from_sympy, coeffs_list))
    for k in lengths:
        all_coeffs.append(coeffs_list[:k])
        coeffs_list = coeffs_list[k:]
    polys = []
    for monoms, coeffs in zip(all_monoms, all_coeffs):
        rep = dict(list(zip(monoms, coeffs)))
        poly = Poly._from_dict(rep, opt)
        polys.append(poly)
    if opt.polys is None:
        opt.polys = bool(_polys)
    return (polys, opt)