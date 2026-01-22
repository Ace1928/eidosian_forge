from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _fwd(subj: Optional[_SubjectType]=None, obj: Optional[_ObjectType]=None, seen: Optional[Set[_SubjectType]]=None) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
    seen.add(subj)
    for s, o in eval_path(graph, (subj, self.path, None)):
        if not obj or o == obj:
            yield (s, o)
        if self.more:
            if o in seen:
                continue
            for s2, o2 in _fwd(o, obj, seen):
                yield (s, o2)