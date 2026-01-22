from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
class NTSerializer(Serializer):
    """
    Serializes RDF graphs to NTriples format.
    """

    def __init__(self, store: Graph):
        Serializer.__init__(self, store)

    def serialize(self, stream: IO[bytes], base: Optional[str]=None, encoding: Optional[str]='utf-8', **args) -> None:
        if base is not None:
            warnings.warn('NTSerializer does not support base.')
        if encoding != 'utf-8':
            warnings.warn(f'NTSerializer always uses UTF-8 encoding. Given encoding was: {encoding}')
        for triple in self.store:
            stream.write(_nt_row(triple).encode())