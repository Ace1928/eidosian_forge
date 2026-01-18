import re
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, cast
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from w3lib.url import *
from w3lib.url import _safe_chars, _unquotepath  # noqa: F401
from scrapy.utils.python import to_unicode
def url_has_any_extension(url: UrlT, extensions: Iterable[str]) -> bool:
    """Return True if the url ends with one of the extensions provided"""
    lowercase_path = parse_url(url).path.lower()
    return any((lowercase_path.endswith(ext) for ext in extensions))