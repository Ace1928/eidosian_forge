from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _prep_sources(self, base: Optional[str], inputs: List[Any], sources: List[Any], referenced_contexts: Set[str], in_source_url: Optional[str]=None):
    for source in inputs:
        source_url = in_source_url
        new_base = base
        if isinstance(source, str):
            source_url = source
            source_doc_base = base or self.doc_base
            new_ctx = self._fetch_context(source, source_doc_base, referenced_contexts)
            if new_ctx is None:
                continue
            else:
                if base:
                    if TYPE_CHECKING:
                        assert source_doc_base is not None
                    new_base = urljoin(source_doc_base, source_url)
                source = new_ctx
        if isinstance(source, dict):
            if CONTEXT in source:
                source = source[CONTEXT]
                source = source if isinstance(source, list) else [source]
        if isinstance(source, list):
            self._prep_sources(new_base, source, sources, referenced_contexts, source_url)
        else:
            sources.append((source_url, source))