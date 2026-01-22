from __future__ import annotations
import codecs
import re
from io import BytesIO, StringIO, TextIOBase
from typing import (
from rdflib.compat import _string_escape_map, decodeUnicodeEscape
from rdflib.exceptions import ParserError as ParseError
from rdflib.parser import InputSource, Parser
from rdflib.term import BNode as bNode
from rdflib.term import Literal
from rdflib.term import URIRef
from rdflib.term import URIRef as URI
class NTParser(Parser):
    """parser for the ntriples format, often stored with the .nt extension

    See http://www.w3.org/TR/rdf-testcases/#ntriples"""
    __slots__ = ()

    @classmethod
    def parse(cls, source: InputSource, sink: 'Graph', **kwargs: Any) -> None:
        """
        Parse the NT format

        :type source: `rdflib.parser.InputSource`
        :param source: the source of NT-formatted data
        :type sink: `rdflib.graph.Graph`
        :param sink: where to send parsed triples
        :param kwargs: Additional arguments to pass to `.W3CNTriplesParser.parse`
        """
        f: Union[TextIO, IO[bytes], codecs.StreamReader]
        f = source.getCharacterStream()
        if not f:
            b = source.getByteStream()
            if isinstance(b, TextIOBase):
                f = b
            else:
                f = codecs.getreader('utf-8')(b)
        parser = W3CNTriplesParser(NTGraphSink(sink))
        parser.parse(f, **kwargs)
        f.close()