from __future__ import annotations
import collections
from typing import (
from rdflib.plugins.sparql.operators import EBV
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _ebv(expr: Union[Literal, Variable, Expr], ctx: FrozenDict) -> bool:
    """
    Return true/false for the given expr
    Either the expr is itself true/false
    or evaluates to something, with the given ctx

    an error is false
    """
    try:
        return EBV(expr)
    except SPARQLError:
        pass
    if isinstance(expr, Expr):
        try:
            return EBV(expr.eval(ctx))
        except SPARQLError:
            return False
    elif isinstance(expr, CompValue):
        raise Exception('Weird - filter got a CompValue without evalfn! %r' % expr)
    elif isinstance(expr, Variable):
        try:
            return EBV(ctx[expr])
        except:
            return False
    return False