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
def _unquote_partial(value: str) -> str:
    parts = iter(pattern.split(value))
    out = []
    for part in parts:
        out.append(unquote(part, 'utf-8', 'werkzeug.url_quote'))
        out.append(next(parts, ''))
    return ''.join(out)