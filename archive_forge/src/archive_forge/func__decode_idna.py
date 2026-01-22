from __future__ import annotations
import codecs
import re
import typing as t
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from .datastructures import iter_multi_items
def _decode_idna(domain: str) -> str:
    try:
        data = domain.encode('ascii')
    except UnicodeEncodeError:
        return domain
    try:
        return data.decode('idna')
    except UnicodeDecodeError:
        pass
    parts = []
    for part in data.split(b'.'):
        try:
            parts.append(part.decode('idna'))
        except UnicodeDecodeError:
            parts.append(part.decode('ascii'))
    return '.'.join(parts)