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
def translatePName(p: typing.Union[CompValue, str], prologue: Prologue) -> Optional[Identifier]:
    """
    Expand prefixed/relative URIs
    """
    if isinstance(p, CompValue):
        if p.name == 'pname':
            return prologue.absolutize(p)
        if p.name == 'literal':
            return Literal(p.string, lang=p.lang, datatype=prologue.absolutize(p.datatype))
    elif isinstance(p, URIRef):
        return prologue.absolutize(p)