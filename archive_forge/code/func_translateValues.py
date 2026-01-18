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
def translateValues(v: CompValue) -> typing.Union[List[Dict[Variable, str]], CompValue]:
    res: List[Dict[Variable, str]] = []
    if not v.var:
        return res
    if not v.value:
        return res
    if not isinstance(v.value[0], list):
        for val in v.value:
            res.append({v.var[0]: val})
    else:
        for vals in v.value:
            res.append(dict(zip(v.var, vals)))
    return Values(res)