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
def _unicodeEscape(self, argstr: str, i: int, startline: int, reg: Pattern[str], n: int, prefix: str) -> Tuple[int, str]:
    if len(argstr) < i + n:
        raise BadSyntax(self._thisDoc, startline, argstr, i, 'unterminated string literal(3)')
    try:
        return (i + n, reg.sub(unicodeExpand, '\\' + prefix + argstr[i:i + n]))
    except Exception:
        raise BadSyntax(self._thisDoc, startline, argstr, i, 'bad string literal hex escape: ' + argstr[i:i + n])