import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
def to_raw_value(self, graph, s, o, nodemap):
    context = self.context
    coll = self.to_collection(graph, o)
    if coll is not None:
        coll = [self.to_raw_value(graph, s, lo, nodemap) for lo in self.to_collection(graph, o)]
        return {context.list_key: coll}
    elif isinstance(o, BNode):
        embed = False
        onode = self.process_subject(graph, o, nodemap)
        if onode:
            if embed and (not any((s2 for s2 in graph.subjects(None, o) if s2 != s))):
                return onode
            else:
                nodemap[onode[context.id_key]] = onode
        return {context.id_key: o.n3()}
    elif isinstance(o, URIRef):
        return {context.id_key: context.shrink_iri(o)}
    elif isinstance(o, Literal):
        native = self.use_native_types and o.datatype in PLAIN_LITERAL_TYPES
        if native:
            v = o.toPython()
        else:
            v = str(o)
        if o.datatype:
            if native:
                if self.context.active:
                    return v
                else:
                    return {context.value_key: v}
            return {context.type_key: context.to_symbol(o.datatype), context.value_key: v}
        elif o.language and o.language != context.language:
            return {context.lang_key: o.language, context.value_key: v}
        elif not context.active or (context.language and (not o.language)):
            return {context.value_key: v}
        else:
            return v