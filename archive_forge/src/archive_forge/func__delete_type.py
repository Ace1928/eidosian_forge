import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
@TermDeletionHelper(RDF.type)
def _delete_type(self):
    """
        >>> g = Graph()
        >>> b = Individual(OWL.Restriction, g)
        >>> b.type = RDFS.Resource
        >>> len(list(b.type))
        1
        >>> del b.type
        >>> len(list(b.type))
        0
        """
    pass