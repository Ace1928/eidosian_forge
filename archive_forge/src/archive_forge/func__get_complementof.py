import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def _get_complementof(self):
    comp = list(self.graph.objects(subject=self.identifier, predicate=OWL.complementOf))
    if not comp:
        return None
    elif len(comp) == 1:
        return Class(comp[0], graph=self.graph)
    else:
        raise Exception(len(comp))