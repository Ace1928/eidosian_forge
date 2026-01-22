from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def _eval_seq_bw(paths: List[Union[Path, URIRef]], subj: Optional[_SubjectType], obj: _ObjectType) -> Generator[Tuple[_SubjectType, _ObjectType], None, None]:
    if paths[:-1]:
        for s, o in eval_path(graph, (None, paths[-1], obj)):
            for r in _eval_seq(paths[:-1], subj, s):
                yield (r[0], o)
    else:
        for s, o in eval_path(graph, (subj, paths[0], obj)):
            yield (s, o)