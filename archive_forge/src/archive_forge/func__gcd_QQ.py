from __future__ import annotations
from typing import Any
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.numbers import igcd, oo
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import (Domain as DomainOpt,
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
def _gcd_QQ(self, g):
    f = self
    ring = f.ring
    new_ring = ring.clone(domain=ring.domain.get_ring())
    cf, f = f.clear_denoms()
    cg, g = g.clear_denoms()
    f = f.set_ring(new_ring)
    g = g.set_ring(new_ring)
    h, cff, cfg = f._gcd_ZZ(g)
    h = h.set_ring(ring)
    c, h = (h.LC, h.monic())
    cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
    cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))
    return (h, cff, cfg)