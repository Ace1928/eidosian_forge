import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
def add_to_node(self, graph, s, p, o, s_node, nodemap):
    context = self.context
    if isinstance(o, Literal):
        datatype = str(o.datatype) if o.datatype else None
        language = o.language
        term = context.find_term(str(p), datatype, language=language)
    else:
        containers = [LIST, None] if graph.value(o, RDF.first) else [None]
        for container in containers:
            for coercion in (ID, VOCAB, UNDEF):
                term = context.find_term(str(p), coercion, container)
                if term:
                    break
            if term:
                break
    node = None
    use_set = not context.active
    if term:
        p_key = term.name
        if term.type:
            node = self.type_coerce(o, term.type)
        elif term.language and o.language == term.language:
            node = str(o)
        elif context.language and (term.language is None and o.language is None):
            node = str(o)
        if LIST in term.container:
            node = [self.type_coerce(v, term.type) or self.to_raw_value(graph, s, v, nodemap) for v in self.to_collection(graph, o)]
        elif LANG in term.container and language:
            value = s_node.setdefault(p_key, {})
            values = value.get(language)
            node = str(o)
            if values or SET in term.container:
                if not isinstance(values, list):
                    value[language] = values = [values]
                values.append(node)
            else:
                value[language] = node
            return
        elif SET in term.container:
            use_set = True
    else:
        p_key = context.to_symbol(p)
        key_term = context.terms.get(p_key)
        if key_term and (key_term.type or key_term.container):
            p_key = p
        if not term and p == RDF.type and (not self.use_rdf_type):
            if isinstance(o, URIRef):
                node = context.to_symbol(o)
            p_key = context.type_key
    if node is None:
        node = self.to_raw_value(graph, s, o, nodemap)
    value = s_node.get(p_key)
    if value:
        if not isinstance(value, list):
            value = [value]
        value.append(node)
    elif use_set:
        value = [node]
    else:
        value = node
    s_node[p_key] = value