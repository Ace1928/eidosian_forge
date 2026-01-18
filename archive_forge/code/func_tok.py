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
def tok(self, tok: str, argstr: str, i: int, colon: bool=False) -> int:
    """Check for keyword.  Space must have been stripped on entry and
        we must not be at end of file.

        if colon, then keyword followed by colon is ok
        (@prefix:<blah> is ok, rdf:type shortcut a must be followed by ws)
        """
    assert tok[0] not in _notNameChars
    if argstr[i] == '@':
        i += 1
    elif tok not in self.keywords:
        return -1
    i_plus_len_tok = i + len(tok)
    if argstr[i:i_plus_len_tok] == tok and argstr[i_plus_len_tok] in _notKeywordsChars or (colon and argstr[i_plus_len_tok] == ':'):
        return i_plus_len_tok
    else:
        return -1