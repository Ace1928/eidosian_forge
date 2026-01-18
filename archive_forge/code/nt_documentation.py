from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal

    Do unicode char replaces as defined in https://www.w3.org/TR/2004/REC-rdf-testcases-20040210/#ntrip_strings
    