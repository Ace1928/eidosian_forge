import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def write_start_result(self) -> None:
    self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'result'), 'result', AttributesNSImpl({}, {}))
    self._resultStarted = True