import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
@TermDeletionHelper(RDFS.label)
def _delete_label(self):
    """
        >>> g = Graph()
        >>> b = Individual(OWL.Restriction,g)
        >>> b.label = Literal('boo')
        >>> len(list(b.label))
        1
        >>> del b.label
        >>> len(list(b.label))
        0
        """
    pass