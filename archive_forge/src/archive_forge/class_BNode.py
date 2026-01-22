import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
class BNode(IdentifiedNode):
    """
    RDF 1.1's Blank Nodes Section: https://www.w3.org/TR/rdf11-concepts/#section-blank-nodes

    Blank Nodes are local identifiers for unnamed nodes in RDF graphs that are used in
    some concrete RDF syntaxes or RDF store implementations. They are always locally
    scoped to the file or RDF store, and are not persistent or portable identifiers for
    blank nodes. The identifiers for Blank Nodes are not part of the RDF abstract
    syntax, but are entirely dependent on particular concrete syntax or implementation
    (such as Turtle, JSON-LD).

    ---

    RDFLib's ``BNode`` class makes unique IDs for all the Blank Nodes in a Graph but you
    should *never* expect, or reply on, BNodes' IDs to match across graphs, or even for
    multiple copies of the same graph, if they are regenerated from some non-RDFLib
    source, such as loading from RDF data.
    """
    __slots__ = ()

    def __new__(cls, value: Optional[str]=None, _sn_gen: Callable[[], str]=_serial_number_generator(), _prefix: str=_unique_id()) -> 'BNode':
        """
        # only store implementations should pass in a value
        """
        if value is None:
            node_id = _sn_gen()
            value = '%s%s' % (_prefix, node_id)
        else:
            pass
        return Identifier.__new__(cls, value)

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        return '_:%s' % self

    def __reduce__(self) -> Tuple[Type['BNode'], Tuple[str]]:
        return (BNode, (str(self),))

    def __repr__(self) -> str:
        if self.__class__ is BNode:
            clsName = 'rdflib.term.BNode'
        else:
            clsName = self.__class__.__name__
        return "%s('%s')" % (clsName, str(self))

    def skolemize(self, authority: Optional[str]=None, basepath: Optional[str]=None) -> URIRef:
        """Create a URIRef "skolem" representation of the BNode, in accordance
        with http://www.w3.org/TR/rdf11-concepts/#section-skolemization

        .. versionadded:: 4.0
        """
        if authority is None:
            authority = _SKOLEM_DEFAULT_AUTHORITY
        if basepath is None:
            basepath = rdflib_skolem_genid
        skolem = '%s%s' % (basepath, str(self))
        return URIRef(urljoin(authority, skolem))