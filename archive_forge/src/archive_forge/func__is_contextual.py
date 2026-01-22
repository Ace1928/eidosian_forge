import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def _is_contextual(self, graph: Optional[Union['Graph', 'str']]) -> bool:
    """Returns `True` if the "GRAPH" keyword must appear
        in the final SPARQL query sent to the endpoint.
        """
    if not self.context_aware or graph is None:
        return False
    if isinstance(graph, str):
        return graph != '__UNION__'
    else:
        return graph.identifier != DATASET_DEFAULT_GRAPH_ID