from typing import IO, Optional
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import Namespace
from rdflib.plugins.serializers.xmlwriter import XMLWriter
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def _writeGraph(self, graph):
    self.writer.push(TRIXNS['graph'])
    if graph.base:
        self.writer.attribute('http://www.w3.org/XML/1998/namespacebase', graph.base)
    if isinstance(graph.identifier, URIRef):
        self.writer.element(TRIXNS['uri'], content=str(graph.identifier))
    for triple in graph.triples((None, None, None)):
        self._writeTriple(triple)
    self.writer.pop()