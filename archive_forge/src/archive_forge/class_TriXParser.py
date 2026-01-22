from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from xml.sax import handler, make_parser
from xml.sax.handler import ErrorHandler
from rdflib.exceptions import ParserError
from rdflib.graph import Graph
from rdflib.namespace import Namespace
from rdflib.parser import InputSource, Parser
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Literal, URIRef
class TriXParser(Parser):
    """A parser for TriX. See http://sw.nokia.com/trix/"""

    def __init__(self):
        pass

    def parse(self, source: InputSource, sink: Graph, **args: Any) -> None:
        assert sink.store.context_aware, 'TriXParser must be given a context aware store.'
        self._parser = create_parser(sink.store)
        content_handler = self._parser.getContentHandler()
        preserve_bnode_ids = args.get('preserve_bnode_ids', None)
        if preserve_bnode_ids is not None:
            content_handler.preserve_bnode_ids = preserve_bnode_ids
        self._parser.parse(source)