from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
def _nt_row(triple: _TripleType) -> str:
    if isinstance(triple[2], Literal):
        return '%s %s %s .\n' % (triple[0].n3(), triple[1].n3(), _quoteLiteral(triple[2]))
    else:
        return '%s %s %s .\n' % (triple[0].n3(), triple[1].n3(), triple[2].n3())