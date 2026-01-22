from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _subcontext(self, source: Any, propagate: bool) -> 'Context':
    ctx = Context(version=self.version)
    ctx.propagate = propagate
    ctx.parent = self
    ctx.language = self.language
    ctx.vocab = self.vocab
    ctx.base = self.base
    ctx.doc_base = self.doc_base
    ctx._alias = {k: l[:] for k, l in self._alias.items()}
    ctx.terms = self.terms.copy()
    ctx._lookup = self._lookup.copy()
    ctx._prefixes = self._prefixes.copy()
    ctx._context_cache = self._context_cache
    ctx.load(source)
    return ctx