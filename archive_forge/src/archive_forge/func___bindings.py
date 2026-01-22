import xml.dom.minidom
from typing import IO, Dict, Optional, Set
from xml.sax.saxutils import escape, quoteattr
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import RDF, RDFS, Namespace  # , split_uri
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.plugins.serializers.xmlwriter import XMLWriter
from rdflib.serializer import Serializer
from rdflib.term import BNode, IdentifiedNode, Identifier, Literal, Node, URIRef
from rdflib.util import first, more_than
from .xmlwriter import ESCAPE_ENTITIES
def __bindings(self):
    store = self.store
    nm = store.namespace_manager
    bindings = {}
    for predicate in set(store.predicates()):
        prefix, namespace, name = nm.compute_qname_strict(predicate)
        bindings[prefix] = URIRef(namespace)
    RDFNS = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    if 'rdf' in bindings:
        assert bindings['rdf'] == RDFNS
    else:
        bindings['rdf'] = RDFNS
    for prefix, namespace in bindings.items():
        yield (prefix, namespace)