from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
def _quote_encode(l_: str) -> str:
    return '"%s"' % l_.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"').replace('\r', '\\r')