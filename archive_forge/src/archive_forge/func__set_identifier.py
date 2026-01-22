import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_identifier(self, i):
    assert i
    if i != self.__identifier:
        oldstatements_out = [(p, o) for s, p, o in self.graph.triples((self.__identifier, None, None))]
        oldstatements_in = [(s, p) for s, p, o in self.graph.triples((None, None, self.__identifier))]
        for p1, o1 in oldstatements_out:
            self.graph.remove((self.__identifier, p1, o1))
        for s1, p1 in oldstatements_in:
            self.graph.remove((s1, p1, self.__identifier))
        self.__identifier = i
        self.graph.addN([(i, p1, o1, self.graph) for p1, o1 in oldstatements_out])
        self.graph.addN([(s1, p1, i, self.graph) for s1, p1 in oldstatements_in])
    if not isinstance(i, BNode):
        try:
            prefix, uri, localname = self.graph.compute_qname(i)
            self.qname = ':'.join([prefix, localname])
        except Exception:
            pass