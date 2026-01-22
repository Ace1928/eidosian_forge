import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class BooleanClassExtentHelper:
    """
    >>> testGraph = Graph()
    >>> Individual.factoryGraph = testGraph
    >>> EX = Namespace("http://example.com/")
    >>> testGraph.bind("ex", EX, override=False)
    >>> fire = Class(EX.Fire)
    >>> water = Class(EX.Water)
    >>> testClass = BooleanClass(members=[fire, water])
    >>> testClass2 = BooleanClass(
    ...     operator=OWL.unionOf, members=[fire, water])
    >>> for c in BooleanClass.getIntersections():
    ...     print(c)  # doctest: +SKIP
    ( ex:Fire AND ex:Water )
    >>> for c in BooleanClass.getUnions():
    ...     print(c) #doctest: +SKIP
    ( ex:Fire OR ex:Water )
    """

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, f):

        def _getExtent():
            for c in Individual.factoryGraph.subjects(self.operator):
                yield BooleanClass(c, operator=self.operator)
        return _getExtent