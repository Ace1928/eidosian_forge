from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _read_term(self, source: Dict[str, Any], name: str, dfn: Union[Dict[str, Any], str], protected: bool=False) -> None:
    idref = None
    if isinstance(dfn, dict):
        rev = dfn.get(REV)
        protected = dfn.get(PROTECTED, protected)
        coercion = dfn.get(TYPE, UNDEF)
        if coercion and coercion not in (ID, TYPE, VOCAB):
            coercion = self._rec_expand(source, coercion)
        idref = rev or dfn.get(ID, UNDEF)
        if idref == TYPE:
            idref = str(RDF.type)
            coercion = VOCAB
        elif idref is not UNDEF:
            idref = self._rec_expand(source, idref)
        elif ':' in name:
            idref = self._rec_expand(source, name)
        elif self.vocab:
            idref = self.vocab + name
        context = dfn.get(CONTEXT, UNDEF)
        self.add_term(name, idref, coercion, dfn.get(CONTAINER, UNDEF), dfn.get(INDEX, UNDEF), dfn.get(LANG, UNDEF), bool(rev), context, dfn.get(PREFIX), protected=protected)
    else:
        if isinstance(dfn, str):
            if not self._accept_term(dfn):
                return
            idref = self._rec_expand(source, dfn)
        self.add_term(name, idref, protected=protected)
    if idref in NODE_KEYS:
        self._alias.setdefault(idref, []).append(name)