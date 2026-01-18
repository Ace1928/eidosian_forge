import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
def to_collection(self, graph, l_):
    if l_ != RDF.nil and (not graph.value(l_, RDF.first)):
        return None
    list_nodes = []
    chain = set([l_])
    while l_:
        if l_ == RDF.nil:
            return list_nodes
        if isinstance(l_, URIRef):
            return None
        first, rest = (None, None)
        for p, o in graph.predicate_objects(l_):
            if not first and p == RDF.first:
                first = o
            elif not rest and p == RDF.rest:
                rest = o
            elif p != RDF.type or o != RDF.List:
                return None
        list_nodes.append(first)
        l_ = rest
        if l_ in chain:
            return None
        chain.add(l_)