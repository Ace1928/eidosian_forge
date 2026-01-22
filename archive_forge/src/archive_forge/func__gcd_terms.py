from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
def _gcd_terms(terms, isprimitive=False, fraction=True):
    """Helper function for :func:`gcd_terms`.

    Parameters
    ==========

    isprimitive : boolean, optional
        If ``isprimitive`` is True then the call to primitive
        for an Add will be skipped. This is useful when the
        content has already been extracted.

    fraction : boolean, optional
        If ``fraction`` is True then the expression will appear over a common
        denominator, the lcm of all term denominators.
    """
    if isinstance(terms, Basic) and (not isinstance(terms, Tuple)):
        terms = Add.make_args(terms)
    terms = list(map(Term, [t for t in terms if t]))
    if len(terms) == 0:
        return (S.Zero, S.Zero, S.One)
    if len(terms) == 1:
        cont = terms[0].coeff
        numer = terms[0].numer.as_expr()
        denom = terms[0].denom.as_expr()
    else:
        cont = terms[0]
        for term in terms[1:]:
            cont = cont.gcd(term)
        for i, term in enumerate(terms):
            terms[i] = term.quo(cont)
        if fraction:
            denom = terms[0].denom
            for term in terms[1:]:
                denom = denom.lcm(term.denom)
            numers = []
            for term in terms:
                numer = term.numer.mul(denom.quo(term.denom))
                numers.append(term.coeff * numer.as_expr())
        else:
            numers = [t.as_expr() for t in terms]
            denom = Term(S.One).numer
        cont = cont.as_expr()
        numer = Add(*numers)
        denom = denom.as_expr()
    if not isprimitive and numer.is_Add:
        _cont, numer = numer.primitive()
        cont *= _cont
    return (cont, numer, denom)