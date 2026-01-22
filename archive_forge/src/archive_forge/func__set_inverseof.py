import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_inverseof(self, other):
    if not other:
        return
    self.graph.add((self.identifier, OWL.inverseOf, classOrIdentifier(other)))