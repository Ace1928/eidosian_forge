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
def _get_nested_id(self, context: Context, node: Dict[str, Any]) -> Optional[str]:
    for key, obj in node.items():
        if context.version >= 1.1 and key in context.get_keys(NEST):
            term = context.terms.get(key)
            if term and term.id is None:
                continue
            objs = obj if isinstance(obj, list) else [obj]
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                id_val = context.get_id(obj)
                if not id_val:
                    subcontext = context.get_context_for_term(context.terms.get(key))
                    id_val = self._get_nested_id(subcontext, obj)
                if isinstance(id_val, str):
                    return id_val