import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _set_comment(self, comment):
    if not comment:
        return
    if isinstance(comment, Identifier):
        self.graph.add((self.identifier, RDFS.comment, comment))
    else:
        for c in comment:
            self.graph.add((self.identifier, RDFS.comment, c))