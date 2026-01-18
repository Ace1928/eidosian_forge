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
def to_rdf(data: Any, dataset: Graph, base: Optional[str]=None, context_data: Optional[bool]=None, version: Optional[float]=None, generalized_rdf: bool=False, allow_lists_of_lists: Optional[bool]=None):
    context = Context(base=base, version=version)
    if context_data:
        context.load(context_data)
    parser = Parser(generalized_rdf=generalized_rdf, allow_lists_of_lists=allow_lists_of_lists)
    return parser.parse(data, context, dataset)