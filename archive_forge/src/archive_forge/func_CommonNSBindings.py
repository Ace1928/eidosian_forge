import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def CommonNSBindings(graph, additionalNS=None):
    """
    Takes a graph and binds the common namespaces (rdf,rdfs, & owl)
    """
    additional_ns = {} if additionalNS is None else additionalNS
    namespace_manager = NamespaceManager(graph)
    namespace_manager.bind('rdfs', RDFS)
    namespace_manager.bind('rdf', RDF)
    namespace_manager.bind('owl', OWL)
    for prefix, uri in list(additional_ns.items()):
        namespace_manager.bind(prefix, uri, override=False)
    graph.namespace_manager = namespace_manager