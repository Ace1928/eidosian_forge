from __future__ import annotations
from decimal import Decimal
from typing import (
from rdflib.namespace import XSD
from rdflib.plugins.sparql.datatypes import type_promotion
from rdflib.plugins.sparql.evalutils import _eval, _val
from rdflib.plugins.sparql.operators import numeric
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenBindings, NotBoundError, SPARQLTypeError
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class GroupConcat(Accumulator):
    value: List[Literal]

    def __init__(self, aggregation: CompValue):
        super(GroupConcat, self).__init__(aggregation)
        self.value = []
        if aggregation.separator is None:
            self.separator = ' '
        else:
            self.separator = aggregation.separator

    def update(self, row: FrozenBindings, aggregator: 'Aggregator') -> None:
        try:
            value = _eval(self.expr, row)
            if isinstance(value, NotBoundError):
                return
            self.value.append(value)
            if self.distinct:
                self.seen.add(value)
        except NotBoundError:
            pass

    def get_value(self) -> Literal:
        return Literal(self.separator.join((str(v) for v in self.value)))