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
def _warn_for_remote_retrieve(uri: str):
    from urllib.request import Request, urlopen
    headers = {'User-Agent': 'python-jsonschema (deprecated $ref resolution)'}
    request = Request(uri, headers=headers)
    with urlopen(request) as response:
        warnings.warn('Automatically retrieving remote references can be a security vulnerability and is discouraged by the JSON Schema specifications. Relying on this behavior is deprecated and will shortly become an error. If you are sure you want to remotely retrieve your reference and that it is safe to do so, you can find instructions for doing so via referencing.Registry in the referencing documentation (https://referencing.readthedocs.org).', DeprecationWarning, stacklevel=9)
        return referencing.Resource.from_contents(json.load(response), default_specification=referencing.jsonschema.DRAFT202012)