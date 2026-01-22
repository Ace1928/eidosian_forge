from __future__ import annotations
import collections
from typing import (
from rdflib.plugins.sparql.operators import EBV
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _minus(a: Iterable[_FrozenDictT], b: Iterable[_FrozenDictT]) -> Generator[_FrozenDictT, None, None]:
    for x in a:
        if all((not x.compatible(y) or x.disjointDomain(y) for y in b)):
            yield x