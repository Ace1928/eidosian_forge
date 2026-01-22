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
@staticmethod
def _to_typed_json_value(value: Any) -> Dict[str, str]:
    return {TYPE: URIRef('%sJSON' % str(RDF)), VALUE: json.dumps(value, separators=(',', ':'), sort_keys=True, ensure_ascii=False)}