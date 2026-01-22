from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _fetch_context(self, source: str, base: Optional[str], referenced_contexts: Set[str]):
    source_url = urljoin(base, source)
    if source_url in referenced_contexts:
        raise RECURSIVE_CONTEXT_INCLUSION
    referenced_contexts.add(source_url)
    if source_url in self._context_cache:
        return self._context_cache[source_url]
    source = source_to_json(source_url)
    if source and CONTEXT not in source:
        raise INVALID_REMOTE_CONTEXT
    self._context_cache[source_url] = source
    return source