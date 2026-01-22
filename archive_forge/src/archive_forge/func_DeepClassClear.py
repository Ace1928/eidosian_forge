import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def DeepClassClear(class_to_prune):
    """
    Recursively clear the given class, continuing
    where any related class is an anonymous class

    >>> EX = Namespace("http://example.com/")
    >>> g = Graph()
    >>> g.bind("ex", EX, override=False)
    >>> Individual.factoryGraph = g
    >>> classB = Class(EX.B)
    >>> classC = Class(EX.C)
    >>> classD = Class(EX.D)
    >>> classE = Class(EX.E)
    >>> classF = Class(EX.F)
    >>> anonClass = EX.someProp @ some @ classD
    >>> classF += anonClass
    >>> list(anonClass.subClassOf)
    [Class: ex:F ]
    >>> classA = classE | classF | anonClass
    >>> classB += classA
    >>> classA.equivalentClass = [Class()]
    >>> classB.subClassOf = [EX.someProp @ some @ classC]
    >>> classA
    ( ex:E OR ex:F OR ( ex:someProp SOME ex:D ) )
    >>> DeepClassClear(classA)
    >>> classA
    (  )
    >>> list(anonClass.subClassOf)
    []
    >>> classB
    Class: ex:B SubClassOf: ( ex:someProp SOME ex:C )

    >>> otherClass = classD | anonClass
    >>> otherClass
    ( ex:D OR ( ex:someProp SOME ex:D ) )
    >>> DeepClassClear(otherClass)
    >>> otherClass
    (  )
    >>> otherClass.delete()
    >>> list(g.triples((otherClass.identifier, None, None)))
    []
    """

    def deepClearIfBNode(_class):
        if isinstance(classOrIdentifier(_class), BNode):
            DeepClassClear(_class)
    class_to_prune = CastClass(class_to_prune, Individual.factoryGraph)
    for c in class_to_prune.subClassOf:
        deepClearIfBNode(c)
    class_to_prune.graph.remove((class_to_prune.identifier, RDFS.subClassOf, None))
    for c in class_to_prune.equivalentClass:
        deepClearIfBNode(c)
    class_to_prune.graph.remove((class_to_prune.identifier, OWL.equivalentClass, None))
    inverse_class = class_to_prune.complementOf
    if inverse_class:
        class_to_prune.graph.remove((class_to_prune.identifier, OWL.complementOf, None))
        deepClearIfBNode(inverse_class)
    if isinstance(class_to_prune, BooleanClass):
        for c in class_to_prune:
            deepClearIfBNode(c)
        class_to_prune.clear()
        class_to_prune.graph.remove((class_to_prune.identifier, class_to_prune._operator, None))