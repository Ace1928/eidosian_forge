from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _n3(arg: Union['URIRef', 'Path'], namespace_manager: Optional['NamespaceManager']=None) -> str:
    if isinstance(arg, (SequencePath, AlternativePath)) and len(arg.args) > 1:
        return '(%s)' % arg.n3(namespace_manager)
    return arg.n3(namespace_manager)