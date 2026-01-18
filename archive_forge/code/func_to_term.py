from __future__ import annotations
from calendar import timegm
from os.path import splitext
from time import altzone, gmtime, localtime, time, timezone
from typing import (
from urllib.parse import quote, urlsplit, urlunsplit
import rdflib.graph  # avoid circular dependency
import rdflib.namespace
import rdflib.term
from rdflib.compat import sign
def to_term(s: Optional[str], default: Optional[rdflib.term.Identifier]=None) -> Optional[rdflib.term.Identifier]:
    """
    Creates and returns an Identifier of type corresponding
    to the pattern of the given positional argument string ``s``:

    '' returns the ``default`` keyword argument value or ``None``

    '<s>' returns ``URIRef(s)`` (i.e. without angle brackets)

    '"s"' returns ``Literal(s)`` (i.e. without doublequotes)

    '_s' returns ``BNode(s)`` (i.e. without leading underscore)

    """
    if not s:
        return default
    elif s.startswith('<') and s.endswith('>'):
        return rdflib.term.URIRef(s[1:-1])
    elif s.startswith('"') and s.endswith('"'):
        return rdflib.term.Literal(s[1:-1])
    elif s.startswith('_'):
        return rdflib.term.BNode(s)
    else:
        msg = "Unrecognised term syntax: '%s'" % s
        raise Exception(msg)