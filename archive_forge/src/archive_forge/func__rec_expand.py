from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _rec_expand(self, source: Dict[str, Any], expr: Optional[str], prev: Optional[str]=None) -> Optional[str]:
    if expr == prev or expr in NODE_KEYS:
        return expr
    nxt: Optional[str]
    is_term, pfx, nxt = self._prep_expand(expr)
    if pfx:
        iri = self._get_source_id(source, pfx)
        if iri is None:
            if pfx + ':' == self.vocab:
                return expr
            else:
                term = self.terms.get(pfx)
                if term:
                    iri = term.id
        if iri is None:
            nxt = expr
        else:
            nxt = iri + nxt
    else:
        nxt = self._get_source_id(source, nxt) or nxt
        if ':' not in nxt and self.vocab:
            return self.vocab + nxt
    return self._rec_expand(source, nxt, expr)