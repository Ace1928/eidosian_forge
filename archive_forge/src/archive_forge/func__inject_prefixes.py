import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def _inject_prefixes(self, query: str, extra_bindings: Mapping[str, Any]) -> str:
    bindings = set(list(self.nsBindings.items()) + list(extra_bindings.items()))
    if not bindings:
        return query
    return '\n'.join(['\n'.join(['PREFIX %s: <%s>' % (k, v) for k, v in bindings]), '', query])