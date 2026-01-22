from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import rdflib.parser
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import RDF, XSD
from rdflib.parser import InputSource, URLInputSource
from rdflib.term import BNode, IdentifiedNode, Literal, Node, URIRef
from ..shared.jsonld.context import UNDEF, Context, Term
from ..shared.jsonld.keys import (
from ..shared.jsonld.util import (
def _add_list(self, dataset: Graph, graph: Graph, context: Context, term: Optional[Term], node_list: Any) -> IdentifiedNode:
    if not isinstance(node_list, list):
        node_list = [node_list]
    first_subj = BNode()
    subj, rest = (first_subj, None)
    for node in node_list:
        if node is None:
            continue
        if rest:
            graph.add((subj, RDF.rest, rest))
            subj = rest
        obj = self._to_object(dataset, graph, context, term, node, inlist=True)
        if obj is None:
            continue
        graph.add((subj, RDF.first, obj))
        rest = BNode()
    if rest:
        graph.add((subj, RDF.rest, RDF.nil))
        return first_subj
    else:
        return RDF.nil