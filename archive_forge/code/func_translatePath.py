from __future__ import annotations
import collections
import functools
import operator
import typing
from functools import reduce
from typing import (
from pyparsing import ParseResults
from rdflib.paths import (
from rdflib.plugins.sparql.operators import TrueFilter, and_
from rdflib.plugins.sparql.operators import simplify as simplifyFilters
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Prologue, Query, Update
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def translatePath(p: typing.Union[CompValue, URIRef]) -> Optional['Path']:
    """
    Translate PropertyPath expressions
    """
    if isinstance(p, CompValue):
        if p.name == 'PathAlternative':
            if len(p.part) == 1:
                return p.part[0]
            else:
                return AlternativePath(*p.part)
        elif p.name == 'PathSequence':
            if len(p.part) == 1:
                return p.part[0]
            else:
                return SequencePath(*p.part)
        elif p.name == 'PathElt':
            if not p.mod:
                return p.part
            elif isinstance(p.part, list):
                if len(p.part) != 1:
                    raise Exception('Denkfehler!')
                return MulPath(p.part[0], p.mod)
            else:
                return MulPath(p.part, p.mod)
        elif p.name == 'PathEltOrInverse':
            if isinstance(p.part, list):
                if len(p.part) != 1:
                    raise Exception('Denkfehler!')
                return InvPath(p.part[0])
            else:
                return InvPath(p.part)
        elif p.name == 'PathNegatedPropertySet':
            if isinstance(p.part, list):
                return NegatedPath(AlternativePath(*p.part))
            else:
                return NegatedPath(p.part)