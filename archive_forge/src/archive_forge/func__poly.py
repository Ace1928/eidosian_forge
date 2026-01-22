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
def _poly(expr, opt):
    terms, poly_terms = ([], [])
    for term in Add.make_args(expr):
        factors, poly_factors = ([], [])
        for factor in Mul.make_args(term):
            if factor.is_Add:
                poly_factors.append(_poly(factor, opt))
            elif factor.is_Pow and factor.base.is_Add and factor.exp.is_Integer and (factor.exp >= 0):
                poly_factors.append(_poly(factor.base, opt).pow(factor.exp))
            else:
                factors.append(factor)
        if not poly_factors:
            terms.append(term)
        else:
            product = poly_factors[0]
            for factor in poly_factors[1:]:
                product = product.mul(factor)
            if factors:
                factor = Mul(*factors)
                if factor.is_Number:
                    product = product.mul(factor)
                else:
                    product = product.mul(Poly._from_expr(factor, opt))
            poly_terms.append(product)
    if not poly_terms:
        result = Poly._from_expr(expr, opt)
    else:
        result = poly_terms[0]
        for term in poly_terms[1:]:
            result = result.add(term)
        if terms:
            term = Add(*terms)
            if term.is_Number:
                result = result.add(term)
            else:
                result = result.add(Poly._from_expr(term, opt))
    return result.reorder(*opt.get('gens', ()), **args)