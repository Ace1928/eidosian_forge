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
def _generic_factor_list(expr, gens, args, method):
    """Helper function for :func:`sqf_list` and :func:`factor_list`. """
    options.allowed_flags(args, ['frac', 'polys'])
    opt = options.build_options(gens, args)
    expr = sympify(expr)
    if isinstance(expr, (Expr, Poly)):
        if isinstance(expr, Poly):
            numer, denom = (expr, 1)
        else:
            numer, denom = together(expr).as_numer_denom()
        cp, fp = _symbolic_factor_list(numer, opt, method)
        cq, fq = _symbolic_factor_list(denom, opt, method)
        if fq and (not opt.frac):
            raise PolynomialError('a polynomial expected, got %s' % expr)
        _opt = opt.clone({'expand': True})
        for factors in (fp, fq):
            for i, (f, k) in enumerate(factors):
                if not f.is_Poly:
                    f, _ = _poly_from_expr(f, _opt)
                    factors[i] = (f, k)
        fp = _sorted_factors(fp, method)
        fq = _sorted_factors(fq, method)
        if not opt.polys:
            fp = [(f.as_expr(), k) for f, k in fp]
            fq = [(f.as_expr(), k) for f, k in fq]
        coeff = cp / cq
        if not opt.frac:
            return (coeff, fp)
        else:
            return (coeff, fp, fq)
    else:
        raise PolynomialError('a polynomial expected, got %s' % expr)