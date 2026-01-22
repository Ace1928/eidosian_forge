import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
class JsonLDSerializer(Serializer):

    def __init__(self, store: Graph):
        super(JsonLDSerializer, self).__init__(store)

    def serialize(self, stream: IO[bytes], base: Optional[str]=None, encoding: Optional[str]=None, **kwargs):
        encoding = encoding or 'utf-8'
        if encoding not in ('utf-8', 'utf-16'):
            warnings.warn(f'JSON should be encoded as unicode. Given encoding was: {encoding}')
        context_data = kwargs.get('context')
        use_native_types = (kwargs.get('use_native_types', False),)
        use_rdf_type = kwargs.get('use_rdf_type', False)
        auto_compact = kwargs.get('auto_compact', False)
        indent = kwargs.get('indent', 2)
        separators = kwargs.get('separators', (',', ': '))
        sort_keys = kwargs.get('sort_keys', True)
        ensure_ascii = kwargs.get('ensure_ascii', False)
        obj = from_rdf(self.store, context_data, base, use_native_types, use_rdf_type, auto_compact=auto_compact)
        data = json.dumps(obj, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
        stream.write(data.encode(encoding, 'replace'))