import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class MalformedClassError(MalformedClass):

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg