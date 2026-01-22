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
def _knownTerms(triple: Tuple[Identifier, Identifier, Identifier], varsknown: Set[typing.Union[BNode, Variable]], varscount: Dict[Identifier, int]) -> Tuple[int, int, bool]:
    return (len([x for x in triple if x not in varsknown and isinstance(x, (Variable, BNode))]), -sum((varscount.get(x, 0) for x in triple)), not isinstance(triple[2], Literal))