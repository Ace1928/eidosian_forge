from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def anonymousNode(self, ln: str) -> BNode:
    """Remember or generate a term for one of these _: anonymous nodes"""
    term = self._anonymousNodes.get(ln, None)
    if term is not None:
        return term
    term = self._store.newBlankNode(self._context, why=self._reason2)
    self._anonymousNodes[ln] = term
    return term