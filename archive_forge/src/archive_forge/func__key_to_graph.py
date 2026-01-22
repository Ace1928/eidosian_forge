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
def _key_to_graph(self, dataset: Graph, graph: Graph, context: Context, subj: Node, key: str, obj: Any, reverse: bool=False, no_id: bool=False) -> None:
    if isinstance(obj, list):
        obj_nodes = obj
    else:
        obj_nodes = [obj]
    term = context.terms.get(key)
    if term:
        term_id = term.id
        if term.type == JSON:
            obj_nodes = [self._to_typed_json_value(obj)]
        elif LIST in term.container:
            obj_nodes = [{LIST: obj_nodes}]
        elif isinstance(obj, dict):
            obj_nodes = self._parse_container(context, term, obj)
    else:
        term_id = None
    if TYPE in (key, term_id):
        term = TYPE_TERM
    if GRAPH in (key, term_id):
        if dataset.context_aware and (not no_id):
            if TYPE_CHECKING:
                assert isinstance(dataset, ConjunctiveGraph)
            subgraph = dataset.get_context(subj)
        else:
            subgraph = graph
        for onode in obj_nodes:
            self._add_to_graph(dataset, subgraph, context, onode)
        return
    if SET in (key, term_id):
        for onode in obj_nodes:
            self._add_to_graph(dataset, graph, context, onode)
        return
    if INCLUDED in (key, term_id):
        for onode in obj_nodes:
            self._add_to_graph(dataset, graph, context, onode)
        return
    if context.version >= 1.1 and key in context.get_keys(NEST):
        term = context.terms.get(key)
        if term and term.id is None:
            return
        objs = obj if isinstance(obj, list) else [obj]
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            for nkey, nobj in obj.items():
                if nkey in context.get_keys(ID):
                    continue
                subcontext = context.get_context_for_type(obj)
                self._key_to_graph(dataset, graph, subcontext, subj, nkey, nobj)
        return
    pred_uri = term.id if term else context.expand(key)
    context = context.get_context_for_term(term)
    flattened = []
    for obj in obj_nodes:
        if isinstance(obj, dict):
            objs = context.get_set(obj)
            if objs is not None:
                obj = objs
        if isinstance(obj, list):
            flattened += obj
            continue
        flattened.append(obj)
    obj_nodes = flattened
    if not pred_uri:
        return
    if term and term.reverse:
        reverse = not reverse
    pred: IdentifiedNode
    bid = self._get_bnodeid(pred_uri)
    if bid:
        if not self.generalized_rdf:
            return
        pred = BNode(bid)
    else:
        pred = URIRef(pred_uri)
    for obj_node in obj_nodes:
        obj = self._to_object(dataset, graph, context, term, obj_node)
        if obj is None:
            continue
        if reverse:
            graph.add((obj, pred, subj))
        else:
            graph.add((subj, pred, obj))