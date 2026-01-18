import re
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, cast
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from w3lib.url import *
from w3lib.url import _safe_chars, _unquotepath  # noqa: F401
from scrapy.utils.python import to_unicode
def url_is_from_spider(url: UrlT, spider: Type['Spider']) -> bool:
    """Return True if the url belongs to the given spider"""
    return url_is_from_any_domain(url, [spider.name] + list(getattr(spider, 'allowed_domains', [])))