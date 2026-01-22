import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_range(self, ranges):
    if not ranges:
        return
    if isinstance(ranges, (Individual, Identifier)):
        self.graph.add((self.identifier, RDFS.range, classOrIdentifier(ranges)))
    else:
        for range in ranges:
            self.graph.add((self.identifier, RDFS.range, classOrIdentifier(range)))