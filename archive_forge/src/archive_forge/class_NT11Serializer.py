from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
class NT11Serializer(NTSerializer):
    """
    Serializes RDF graphs to RDF 1.1 NTriples format.

    Exactly like nt - only utf8 encoded.
    """

    def __init__(self, store: Graph):
        Serializer.__init__(self, store)