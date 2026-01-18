import re
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, cast
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from w3lib.url import *
from w3lib.url import _safe_chars, _unquotepath  # noqa: F401
from scrapy.utils.python import to_unicode
def url_is_from_any_domain(url: UrlT, domains: Iterable[str]) -> bool:
    """Return True if the url belongs to any of the given domains"""
    host = parse_url(url).netloc.lower()
    if not host:
        return False
    domains = [d.lower() for d in domains]
    return any((host == d or host.endswith(f'.{d}') for d in domains))