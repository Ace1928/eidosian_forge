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
def translateQuads(quads: CompValue) -> Tuple[List[Tuple[Identifier, Identifier, Identifier]], DefaultDict[str, List[Tuple[Identifier, Identifier, Identifier]]]]:
    if quads.triples:
        alltriples = triples(quads.triples)
    else:
        alltriples = []
    allquads: DefaultDict[str, List[Tuple[Identifier, Identifier, Identifier]]] = collections.defaultdict(list)
    if quads.quadsNotTriples:
        for q in quads.quadsNotTriples:
            if q.triples:
                allquads[q.term] += triples(q.triples)
    return (alltriples, allquads)