from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
class ParamList(Param):
    """
    A shortcut for a Param with isList=True
    """

    def __init__(self, name: str, expr):
        Param.__init__(self, name, expr, True)