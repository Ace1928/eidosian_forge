from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from urllib.parse import urldefrag, urljoin
from xml.sax import handler, make_parser, xmlreader
from xml.sax.handler import ErrorHandler
from xml.sax.saxutils import escape, quoteattr
from rdflib.exceptions import Error, ParserError
from rdflib.graph import Graph
from rdflib.namespace import RDF, is_ncname
from rdflib.parser import InputSource, Parser
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.term import BNode, Identifier, Literal, URIRef
class ElementHandler:
    __slots__ = ['start', 'char', 'end', 'li', 'id', 'base', 'subject', 'predicate', 'object', 'list', 'language', 'datatype', 'declared', 'data']

    def __init__(self):
        self.start = None
        self.char = None
        self.end = None
        self.li = 0
        self.id = None
        self.base = None
        self.subject = None
        self.object = None
        self.list = None
        self.language = None
        self.datatype = None
        self.declared = None
        self.data = None

    def next_li(self):
        self.li += 1
        return RDFVOC['_%s' % self.li]