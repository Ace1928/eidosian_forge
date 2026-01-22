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
def _unify(f, g):
    g = sympify(g)
    if not g.is_Poly:
        try:
            return (f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g)))
        except CoercionFailed:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
    if len(f.gens) != len(g.gens):
        raise UnificationFailed('Cannot unify %s with %s' % (f, g))
    if not (isinstance(f.rep, DMP) and isinstance(g.rep, DMP)):
        raise UnificationFailed('Cannot unify %s with %s' % (f, g))
    cls = f.__class__
    gens = f.gens
    dom = f.rep.dom.unify(g.rep.dom, gens)
    F = f.rep.convert(dom)
    G = g.rep.convert(dom)

    def per(rep, dom=dom, gens=gens, remove=None):
        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]
            if not gens:
                return dom.to_sympy(rep)
        return cls.new(rep, *gens)
    return (dom, per, F, G)