from __future__ import annotations
from typing import (
from rdflib.store import Store
from rdflib.util import _coalesce
def __contexts(self, triple: '_TripleType') -> Generator['_ContextType', None, None]:
    """return a generator for all the non-quoted contexts
        (dereferenced) the encoded triple appears in"""
    return (self.__context_obj_map.get(ctx_str, ctx_str) for ctx_str in self.__get_context_for_triple(triple, skipQuoted=True) if ctx_str is not None)