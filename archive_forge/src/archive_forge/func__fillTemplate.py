from __future__ import annotations
import collections
from typing import (
from rdflib.plugins.sparql.operators import EBV
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def _fillTemplate(template: Iterable[Tuple[Identifier, Identifier, Identifier]], solution: _ContextType) -> Generator[Tuple[Identifier, Identifier, Identifier], None, None]:
    """
    For construct/deleteWhere and friends

    Fill a triple template with instantiated variables
    """
    bnodeMap: DefaultDict[BNode, BNode] = collections.defaultdict(BNode)
    for t in template:
        s, p, o = t
        _s = solution.get(s)
        _p = solution.get(p)
        _o = solution.get(o)
        _s, _p, _o = [bnodeMap[x] if isinstance(x, BNode) else y for x, y in zip(t, (_s, _p, _o))]
        if _s is not None and _p is not None and (_o is not None):
            yield (_s, _p, _o)