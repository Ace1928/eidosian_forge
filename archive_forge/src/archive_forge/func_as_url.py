from __future__ import annotations
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple
from urllib.parse import parse_qsl, quote, unquote, urlparse
from ..log import get_logger
def as_url(scheme, host=None, port=None, user=None, password=None, path=None, query=None, sanitize=False, mask='**'):
    """Generate URL from component parts."""
    parts = [f'{scheme}://']
    if user or password:
        if user:
            parts.append(safequote(user))
        if password:
            if sanitize:
                parts.extend([':', mask] if mask else [':'])
            else:
                parts.extend([':', safequote(password)])
        parts.append('@')
    parts.append(safequote(host) if host else '')
    if port:
        parts.extend([':', port])
    parts.extend(['/', path])
    return ''.join((str(part) for part in parts if part))