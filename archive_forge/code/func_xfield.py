from __future__ import annotations
from typing import Any
from functools import reduce
from operator import add, mul, lt, le, gt, ge
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.constructor import construct_domain
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.rings import PolyElement
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
@public
def xfield(symbols, domain, order=lex):
    """Construct new rational function field returning (field, (x1, ..., xn)). """
    _field = FracField(symbols, domain, order)
    return (_field, _field.gens)