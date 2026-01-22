from __future__ import annotations
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from operator import methodcaller
from typing import TYPE_CHECKING
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import urlopen
from warnings import warn
import contextlib
import json
import reprlib
import warnings
from attrs import define, field, fields
from jsonschema_specifications import REGISTRY as SPECIFICATIONS
from rpds import HashTrieMap
import referencing.exceptions
import referencing.jsonschema
from jsonschema import (
@lru_cache
def _find_in_subschemas(self, url):
    subschemas = self._get_subschemas_cache()['$id']
    if not subschemas:
        return None
    uri, fragment = urldefrag(url)
    for subschema in subschemas:
        id = subschema['$id']
        if not isinstance(id, str):
            continue
        target_uri = self._urljoin_cache(self.resolution_scope, id)
        if target_uri.rstrip('/') == uri.rstrip('/'):
            if fragment:
                subschema = self.resolve_fragment(subschema, fragment)
            self.store[url] = subschema
            return (url, subschema)
    return None