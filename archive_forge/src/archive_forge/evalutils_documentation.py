from __future__ import annotations
import collections
from typing import (
from rdflib.plugins.sparql.operators import EBV
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
utilitity for ordering things