import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def write_binding(self, name: Variable, val: Identifier) -> None:
    assert self._resultStarted
    attr_vals: Dict[Tuple[Optional[str], str], str] = {(None, 'name'): str(name)}
    attr_qnames: Dict[Tuple[Optional[str], str], str] = {(None, 'name'): 'name'}
    self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'binding'), 'binding', AttributesNSImpl(attr_vals, attr_qnames))
    if isinstance(val, URIRef):
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'uri'), 'uri', AttributesNSImpl({}, {}))
        self.writer.characters(val)
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'uri'), 'uri')
    elif isinstance(val, BNode):
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'bnode'), 'bnode', AttributesNSImpl({}, {}))
        self.writer.characters(val)
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'bnode'), 'bnode')
    elif isinstance(val, Literal):
        attr_vals = {}
        attr_qnames = {}
        if val.language:
            attr_vals[XML_NAMESPACE, 'lang'] = val.language
            attr_qnames[XML_NAMESPACE, 'lang'] = 'xml:lang'
        elif val.datatype:
            attr_vals[None, 'datatype'] = val.datatype
            attr_qnames[None, 'datatype'] = 'datatype'
        self.writer.startElementNS((SPARQL_XML_NAMESPACE, 'literal'), 'literal', AttributesNSImpl(attr_vals, attr_qnames))
        self.writer.characters(val)
        self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'literal'), 'literal')
    else:
        raise Exception('Unsupported RDF term: %s' % val)
    self.writer.endElementNS((SPARQL_XML_NAMESPACE, 'binding'), 'binding')