from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def to_symbol(self, iri: str) -> Optional[str]:
    iri = str(iri)
    term = self.find_term(iri)
    if term:
        return term.name
    ns, name = split_iri(iri)
    if ns == self.vocab:
        return name
    pfx = self._prefixes.get(ns)
    if pfx:
        return ':'.join((pfx, name))
    return iri