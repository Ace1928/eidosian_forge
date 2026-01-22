from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def _object_comparator(a, b):
    """
    for nice clean output we sort the objects of triples,
    some of them are literals,
    these are sorted according to the sort order of the underlying python objects
    in py3 not all things are comparable.
    This falls back on comparing string representations when not.
    """
    try:
        if a > b:
            return 1
        if a < b:
            return -1
        return 0
    except TypeError:
        a = str(a)
        b = str(b)
        return (a > b) - (a < b)