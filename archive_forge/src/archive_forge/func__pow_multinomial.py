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
def _pow_multinomial(self, n):
    multinomials = multinomial_coefficients(len(self), n).items()
    monomial_mulpow = self.ring.monomial_mulpow
    zero_monom = self.ring.zero_monom
    terms = self.items()
    zero = self.ring.domain.zero
    poly = self.ring.zero
    for multinomial, multinomial_coeff in multinomials:
        product_monom = zero_monom
        product_coeff = multinomial_coeff
        for exp, (monom, coeff) in zip(multinomial, terms):
            if exp:
                product_monom = monomial_mulpow(product_monom, monom, exp)
                product_coeff *= coeff ** exp
        monom = tuple(product_monom)
        coeff = product_coeff
        coeff = poly.get(monom, zero) + coeff
        if coeff:
            poly[monom] = coeff
        elif monom in poly:
            del poly[monom]
    return poly