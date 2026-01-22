from __future__ import annotations
import collections
import itertools
import json as j
import re
from typing import (
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pyparsing import ParseException
from rdflib.graph import Graph
from rdflib.plugins.sparql import CUSTOM_EVALS, parser
from rdflib.plugins.sparql.aggregates import Aggregator
from rdflib.plugins.sparql.evalutils import (
from rdflib.plugins.sparql.parserutils import CompValue, value
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _yieldBindingsFromServiceCallResult(ctx: QueryContext, r: Dict[str, Dict[str, str]], variables: List[str]) -> Generator[FrozenBindings, None, None]:
    res_dict: Dict[Variable, Identifier] = {}
    for var in variables:
        if var in r and r[var]:
            var_binding = r[var]
            var_type = var_binding['type']
            if var_type == 'uri':
                res_dict[Variable(var)] = URIRef(var_binding['value'])
            elif var_type == 'literal':
                res_dict[Variable(var)] = Literal(var_binding['value'], datatype=var_binding.get('datatype'), lang=var_binding.get('xml:lang'))
            elif var_type == 'typed-literal':
                res_dict[Variable(var)] = Literal(var_binding['value'], datatype=URIRef(var_binding['datatype']))
            elif var_type == 'bnode':
                res_dict[Variable(var)] = BNode(var_binding['value'])
            else:
                raise ValueError(f'invalid type {var_type!r} for variable {var!r}')
    yield FrozenBindings(ctx, res_dict)